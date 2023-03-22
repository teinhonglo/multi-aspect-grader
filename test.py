#!/bin/python
import os
import json
import argparse
import torch
import torchaudio
from transformers import AutoConfig, AutoFeatureExtractor
from transformers import TrainingArguments, Trainer
from datasets import Dataset, Audio, load_metric, load_from_disk

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict

# local import
from utils import make_dataset, load_from_json
#from metrics_np import compute_mse, _accuracy_within_margin
from metrics_np import compute_metrics
from model import AutoGraderModel, AutoGraderPrototypeModel

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
    
    with open(json_path, 'w') as jf:
        json.dump(embed_dict, jf, indent=4)

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
    model_path = os.path.join(args.model_path, "checkpoint-4800")

    # load train_args, model_args
    train_args, model_args = load_from_json(train_conf_path)

    config = torch.load(config_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    if model_args["model_type"] == 'prototype':
        model = AutoGraderPrototypeModel(model_args, config=config).to(device)
        # num_labels, num_prototypes, dim
        prototype = model.get_prototype()
    else:
        model = AutoGraderModel(model_args, config=config).to(device)
    model.load_state_dict(torch.load(model_path+"/pytorch_model.bin", map_location=device))
    model.eval()

    # loading test set
    def preprocess_function(batch):
        audio = batch["audio"]
        # extract features return input_values
        batch["input_values"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["labels"] = batch['label']
        return batch

    test_dataset_path = os.path.dirname(args.test_json) + "/test_dataset"
    if not os.path.exists(test_dataset_path + "/dataset.arrow"):
        te_dataset = make_dataset(args.test_json)
        te_dataset = te_dataset.map(preprocess_function, num_proc=args.nj)
        te_dataset.save_to_disk(test_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(test_dataset_path + "/dataset.arrow"))
        te_dataset = load_from_disk(test_dataset_path)

    # forward
    def predict(batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device=device).unsqueeze(0)
            output = model(input_values, return_dict=True)
            logits = output.logits
            if model_args["model_type"] == 'prototype':
                embed = output.embeds

        if config.problem_type == "single_label_classification":
            pred_ids = torch.argmax(logits, dim=-1) + 1
        else:
            pred_ids = logits

        pred_ids = pred_ids.detach().cpu().numpy().item()
        batch["pred"] = pred_ids

        if model_args["model_type"] == 'prototype':
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
    predictions_file = os.path.join(args.exp_dir, "predictions.txt")
    with open(predictions_file, 'w') as wf:
        for i in range(len(results)):
            wf.write("{} {} {} \n".format( \
                results["id"][i], results["pred"][i], results["label"][i])
            )

    if model_args["model_type"] == 'prototype':
        # write embeds json
        embeds_file = os.path.join(args.exp_dir, "embeds.json")
        embed_json(results, embeds_file)
        # write proto json
        proto_file = os.path.join(args.exp_dir, "protos.json")
        proto_json(prototype, proto_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-json', type=str)
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--model-path', type=str, default="facebook/wav2vec2-base")
    parser.add_argument('--exp-dir', type=str)
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    main(args)
