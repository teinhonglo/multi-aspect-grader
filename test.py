#!/bin/python
import os
import json
import argparse
import torch
from transformers import Wav2Vec2Processor
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset, Audio, load_metric, load_from_disk

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

# local import
from utils import make_dataset, load_from_json
from metrics_np import compute_metrics
from models.baselines import AutoGraderModel, AutoTextGraderModel, AutoAudioTextGraderModel
from models.multi_aspects import AutoMAGraderModel
from models.prototypes import AutoGraderPrototypeModel, AutoGraderPrototypeRegModel

def embed_json(results, json_path):
    embed_dict = defaultdict(dict)

    for i in range(len(results)):
        wavid = results["id"][i]
        embed = results["embed"][i]
        label = results["label"][i]
        pred = results["pred"][i]

        embed_dict[wavid]["embed"] = embed
        embed_dict[wavid]["label"] = label
        embed_dict[wavid]["pred"] = pred
    
    with open(json_path, 'w') as fn:
        json.dump(embed_dict, fn, indent=4)

def proto_json(prototype, json_path):
    proto_dict = defaultdict(list)

    # num_labels, num_prototypes, dim
    prototype = np.array(prototype)
    num_labels = prototype.shape[0]
    num_prototypes = prototype.shape[1]
    for i in range(num_labels):
        for j in range(num_prototypes):
            name = str(i) + "_" + str(j)
            proto_dict[name] = prototype[i][j].tolist()

    with open(json_path, 'w') as jf:
        json.dump(proto_dict, jf, indent=4)

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_conf_path = os.path.join(args.model_path, "train_conf.json")
    config_path = os.path.join(args.model_path, "config.pth")
    text_config_path = os.path.join(args.model_path, "text_config.pth")
    best_model_path = os.path.join(args.model_path, "best")

    # load train_args, model_args
    train_args, model_args = load_from_json(train_conf_path)

    # load config and model
    config = torch.load(config_path)
    text_config = torch.load(text_config_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path)
    text_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model_type = model_args["model_type"]
    
    if model_type == 'prototype':
        model = AutoGraderPrototypeModel(model_args, config=config, text_config=text_config).to(device)
        # num_labels, num_prototypes, dim
        prototype = model.get_prototype()
    elif model_type == 'prototype_reg':
        model = AutoGraderPrototypeRegModel(model_args, config=config, text_config=text_config).to(device)
        # num_labels, num_prototypes, dim
        prototype = model.get_prototype()
    elif model_type == "multi_aspect":
        model = AutoMAGraderModel(model_args, config=config, text_config=text_config).to(device)
    elif model_type == "baseline":
        model = AutoGraderModel(model_args, config=config, text_config=text_config).to(device)
    elif model_type == "baseline_text":
        model = AutoTextGraderModel(model_args, config=config, text_config=text_config).to(device)
    elif model_type == "baseline_audio_text":
        model = AutoAudioTextGraderModel(model_args, config=config, text_config=text_config).to(device)
    else:
    	raise ValueError(f"Invalid model {model_type}")
    model.load_state_dict(torch.load(best_model_path+"/pytorch_model.bin", map_location=device))
    model.eval()

    # loading test set
    def preprocess_function(batch):
        # extract features return input_values
        batch["input_values"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_values[0]
        batch["input_ids"] = batch['text'].lower()
        batch["prompt_input_ids"] = batch["prompt"].lower()
        batch["delivery"] = batch["delivery"]
        batch["language_use"] = batch["language_use"]
        batch["labels"] = batch['label']
        return batch

    # test set
    model_type = model_args["model_type"]
    if model_type in [ "baseline_text" ]:
        model_name = "-".join(model_args["text_model_path"].split("/"))
    else:
        model_name = "-".join(model_args["model_path"].split("/"))
    
    test_basename = os.path.basename(args.test_json).split('.')[0]
    test_basename += "_tts" if model_args["task_type"] == "mdd-tts" else ""
    test_dataset_path = os.path.dirname(args.test_json) + "/{}/{}_dataset".format(model_name,test_basename)
    
    if not os.path.exists(test_dataset_path + "/dataset.arrow"):
        print("[INFO] Loading data from {} ...".format(args.test_json))
        te_dataset = make_dataset(args.test_json, model_args)
        te_dataset = te_dataset.map(preprocess_function, num_proc=args.nj, remove_columns=["audio"])
        te_dataset.save_to_disk(test_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(test_dataset_path + "/dataset.arrow"))
        te_dataset = load_from_disk(test_dataset_path)

    # forward
    def predict(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
            input_ids = text_tokenizer(batch["input_ids"], max_length=256, padding='max_length', truncation=True, return_tensors='pt').input_ids.to(device)
            prompt_input_ids = text_tokenizer(batch["prompt_input_ids"], max_length=256, padding='max_length', truncation=True, return_tensors='pt').input_ids.to(device)
            delivery = torch.tensor([batch["delivery"]]).to(device)
            language_use = torch.tensor([batch["language_use"]]).to(device)
            
            if model_type == "multi_aspect":
                output = model( input_values=input_values, 
                            input_ids=input_ids, 
                            prompt_input_ids=prompt_input_ids, 
                            delivery=delivery,
                            language_use=language_use,
                            return_dict=True)
            elif model_type in ["baseline", "prototype", "prototype_reg", "baseline_text", "baseline_audio_text"]:
                output = model( input_values=input_values, 
                            input_ids=input_ids, 
                            prompt_input_ids=prompt_input_ids, 
                            return_dict=True)
            else:
                raise Exception(f"{model_type} is not defined")

            logits = output.logits
            embed = output.embeds

        if config.problem_type in ["single_label_classification", "cdw_ce_loss", "test_time_adaptation"]:
            pred_ids = torch.argmax(logits, dim=-1) + 1
        else:
            pred_ids = logits

        pred_ids = pred_ids.detach().cpu().numpy().item()
        batch["pred"] = pred_ids

        #if model_args["model_type"] == 'prototype':
        embed = embed.detach().cpu().numpy()
        batch["embed"] = embed # is list not numpy

        return batch

    # output [pred] [label] is list
    results = te_dataset.map(predict)

    # show pred results
    print("predictions:")
    print("{}".format(results["pred"]))
    print("labels:")
    print("{}".format(results["label"]))

    # write predictions
    test_basename = os.path.basename(args.test_json).split('.')[0]
    output_dir = os.path.join(args.exp_dir, test_basename)
    os.makedirs(output_dir, exist_ok=True)
    
    predictions_file = os.path.join(output_dir, "predictions.txt")
    with open(predictions_file, 'w') as wf:
        for i in range(len(results)):
            wf.write("{} {} {} \n".format( \
                results["id"][i], results["pred"][i], results["label"][i])
            )

    # write embeds json
    embeds_file = os.path.join(output_dir, "embeds.json")
    #embed_json(results, embeds_file)
    # write proto json
    #if model_args["model_type"] == 'prototype':
    #    proto_file = os.path.join(args.exp_dir, "protos.json")
    #    proto_json(prototype, proto_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-json', type=str)
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--model-path', type=str, default="facebook/wav2vec2-base")
    parser.add_argument('--exp-dir', type=str)
    parser.add_argument('--test-set', type=str)
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    main(args)
