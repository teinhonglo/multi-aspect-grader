#!/bin/python
import os
import sys
import json
import glob
import argparse
import random
import torch
import torchaudio
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(threshold=100)

# local import
from utils import make_dataset, DataCollatorWithPadding, cal_class_weight, load_from_json, save_to_json
from metrics_np import compute_metrics
from models.baselines import AutoGraderModel, AutoTextGraderModel, AutoAudioTextGraderModel
from models.multi_aspects import AutoMAGraderModel
from models.prototypes import AutoGraderPrototypeModel, AutoGraderPrototypeRegModel


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load train_args, model_args
    training_args = load_from_json(args.train_conf)
    train_args, model_args = training_args[0], training_args[1]
    # save train_args, model_args to exp_dir
    train_conf_path = os.path.join(args.exp_dir, 'train_conf.json')
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    save_to_json(training_args, train_conf_path)
    # show the model_args
    print("[NOTE] Model args ...")
    print(json.dumps(model_args, indent=4))
    assert model_args["problem_type"] == "test_time_adaptation"

    # load the feature extractor of wav2vec2 
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args["model_path"]
    )

    # load the tokenizer of bert
    text_tokenizer = AutoTokenizer.from_pretrained(
        model_args["text_model_path"]
    )

    # save to exp_dir
    feature_extractor.save_pretrained(args.exp_dir)
    text_tokenizer.save_pretrained(args.exp_dir)
    print("[INFO] Save extractor/tokenizer to {} ...".format(args.exp_dir))

    # NOTE: data preprocess
    def preprocess_function(batch):
        # extract features return input_values
        try:
            # wav2vec (ssl features)
            batch["input_values"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_values[0]
        except:
            # whisper
            batch["input_values"] = feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_features[0]
            
        batch["input_ids"] = batch["text"].lower()
        batch["prompt_input_ids"] = batch["prompt"].lower()
        batch["delivery"] = batch["delivery"]
        batch["language_use"] = batch["language_use"]
        batch["labels"] = batch["label"]
        return batch

    model_type = model_args["model_type"]
    if model_type in [ "baseline_text" ]:
        model_name = "-".join(model_args["text_model_path"].split("/"))
    else:
        model_name = "-".join(model_args["model_path"].split("/"))
    # train set
    train_basename = os.path.basename(args.train_json).split('.')[0]
    train_basename += "_tts" if model_args["task_type"] == "mdd-tts" else ""
    train_dataset_path = os.path.dirname(args.train_json) + "/{}/{}_dataset".format(model_name, train_basename)
    
    if not os.path.exists(train_dataset_path + "/dataset.arrow"):
        print("[INFO] Loading data from {} ...".format(args.train_json))
        tr_dataset = make_dataset(args.train_json, model_args)
        tr_dataset = tr_dataset.map(preprocess_function, num_proc=args.nj, remove_columns=["audio"])
        tr_dataset.save_to_disk(train_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(train_dataset_path + "/dataset.arrow"))
        tr_dataset = load_from_disk(train_dataset_path)

    # valid set
    valid_basename = os.path.basename(args.valid_json).split('.')[0]
    valid_basename += "_tts" if model_args["task_type"] == "mdd-tts" else ""
    valid_dataset_path = os.path.dirname(args.valid_json) + "/{}/{}_dataset".format(model_name, valid_basename)
    if not os.path.exists(valid_dataset_path + "/dataset.arrow"):
        print("[INFO] Loading data from {} ...".format(args.valid_json))
        cv_dataset = make_dataset(args.valid_json, model_args)
        cv_dataset = cv_dataset.map(preprocess_function, num_proc=args.nj, remove_columns=["audio"])
        cv_dataset.save_to_disk(valid_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(valid_dataset_path + "/dataset.arrow"))
        cv_dataset = load_from_disk(valid_dataset_path)

    # data collator
    data_collator = DataCollatorWithPadding(
        feature_extractor=feature_extractor, 
        tokenizer=text_tokenizer,
        problem_type=model_args["problem_type"], 
        task_type=model_args["task_type"], 
        padding=True
    )

    # NOTE: class_weight cal from trainset
    if model_args["class_weight_alpha"] != 0:
        assert model_args["problem_type"] == "single_label_classification"
        if "loss_weight_type" in model_args:
            loss_weight_type = model_args["loss_weight_type"]
        else:
            loss_weight_type = 2
            
        print("[INFO] Use class weight alpha {} and loss_weight_type {} ...".format(model_args["class_weight_alpha"], loss_weight_type))
        class_weight = cal_class_weight(tr_dataset['labels'], model_args["num_labels"], \
            alpha=model_args["class_weight_alpha"], loss_weight_type=loss_weight_type).to(device)
    else:
        print("[INFO] No class weight is provide ...")
        class_weight = None

    # NOTE: define model
    print("[INFO] Train a {} model from {} ...".format(model_args["model_type"], model_args["model_path"]))
    model_type = model_args["model_type"]
    if model_type == "prototype":
        model = AutoGraderPrototypeModel(model_args, class_weight=class_weight, pretrained=True)
        if model_args["init_prototypes"]:
            model.init_prototypes(tr_dataset, path=train_dataset_path)
    elif model_type == "prototype_reg":
        model = AutoGraderPrototypeRegModel(model_args, class_weight=class_weight, pretrained=True)
        if model_args["init_prototypes"]:
            model.init_prototypes(tr_dataset, path=train_dataset_path)
    elif model_type == "multi_aspect":
        model = AutoMAGraderModel(model_args, class_weight=class_weight, pretrained=True).to(device)
    elif model_type == "baseline":
        model = AutoGraderModel(model_args, class_weight=class_weight, pretrained=True)
    elif model_type == "baseline_text":
        model = AutoTextGraderModel(model_args, class_weight=class_weight, pretrained=True)
    elif model_type == "baseline_audio_text":
        model = AutoAudioTextGraderModel(model_args, class_weight=class_weight, pretrained=True)
    else:
        raise ValueError(f"Invalid model {model_type}")
    
    if "init_scaler" in model_args and model_args["init_scaler"]:
        model.init_scaler(tr_dataset)
    
    if "pretrained_path" in model_args:
        print("[INFO] Load pretrained model from {} ...".format(model_args["pretrained_path"]))
        best_model_path = model_args["pretrained_path"] + "/best"
        
        # Load pretrained state_dict
        pretrained_state_dict = torch.load(best_model_path + "/pytorch_model.bin", map_location=device)
        
        # Get current model state_dict
        current_state_dict = model.state_dict()

        total_params = sum(p.numel() for p in current_state_dict.values())
        loaded_params = 0

        # Filter parameters with matching shapes
        filtered_state_dict = {
            k: v for k, v in pretrained_state_dict.items() if k in current_state_dict and v.shape == current_state_dict[k].shape
        }

        # Update the model state_dict with the filtered parameters
        missing_keys = [k for k in current_state_dict if k not in filtered_state_dict]
        model.load_state_dict(filtered_state_dict, strict=False)

        # Calculate the number of successfully loaded parameters
        for k, v in filtered_state_dict.items():
            loaded_params += v.numel()

        # Calculate the ratio of successfully loaded parameters
        success_ratio = loaded_params / total_params
        print("[INFO] Success Load Ratio:", success_ratio)  
        
    if model_args["problem_type"] == "test_time_adaptation":
        # for regression
        print("[INFO] Initialize mean and var (regression) ...")
        tr_labels = torch.tensor(tr_dataset['labels'])
        
        def compute_embeddings(batch):
            input_values = torch.as_tensor(batch["input_values"], device=device).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_values=input_values, return_dict=True)
                hidden_states = outputs.embeds
                batch["embed"] = hidden_states
            
            return batch
        # compute all embeddings
        model = model.to(device)
        model.eval()
        tr_result = tr_dataset.map(compute_embeddings)
        tr_vector = torch.tensor(tr_result["embed"])
        print(f"tr_labels {tr_vector.shape}")
        tr_vector_mean = torch.mean(tr_vector, dim=0).squeeze(0)
        tr_vector_var = torch.var(tr_vector, dim=0).squeeze(0)
        print(f"mean {tr_vector_mean.shape} and var {tr_vector_var.shape}")
        model.init_mean_var(tr_vector_mean, tr_vector_var)
 
    # print # of parameters
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('[INFO] Total parameter number is : {:.3f} M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('[INFO] Total trainable parameter number is : {:.3f} M'.format(sum(p.numel() for p in trainables) / 1e6))
    # save model_config
    torch.save(model.config, args.exp_dir + '/config.pth')
    model.config.to_json_file(args.exp_dir + '/config.json')
    torch.save(model.text_config, args.exp_dir + '/text_config.pth')
    model.text_config.to_json_file(args.exp_dir + '/text_config.json')

    # NOTE: define metric
    def calculate_metrics(pred):

        # preds (0-7 to 1-8)
        # NOTE: teemi: 0-8 to 1-9
        preds = pred.predictions
        
        if model_args["problem_type"] in ["single_label_classification", "test_time_adaptation"]:
            preds = np.argmax(preds, axis=1) + 1
        else:
            preds = preds
        # labels
        labels = pred.label_ids

        print("\n\n")
        print("predictions:")
        print(f"{preds}")
        print("labels:")
        print(f"{labels}")
        print("\n\n")

        # metrics
        total_losses = {}
        compute_metrics(total_losses, np.array(preds), np.array(labels), bins=args.bins)
        return total_losses

    # NOTE: define training args
    #train_args = load_from_json(args.train_conf)
    training_args = TrainingArguments(
        output_dir=args.exp_dir,
        group_by_length=True,
        fp16=True,
        load_best_model_at_end=True,
        **train_args
    )

    if model_args["problem_type"] == "test_time_adaptation":
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=calculate_metrics,
            train_dataset=cv_dataset,
            eval_dataset=cv_dataset,
            tokenizer=feature_extractor,
        )
    else:
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=calculate_metrics,
            train_dataset=tr_dataset,
            eval_dataset=cv_dataset,
            tokenizer=feature_extractor,
        )

    if glob.glob(os.path.join(args.exp_dir, 'checkpoint*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # save the best model
    best_path = os.path.join(args.exp_dir, 'best')
    trainer.save_model(best_path)
    
    for loss in [ "loss" ]:
        train_losses = [log[loss] for log in trainer.state.log_history if loss in log]
        eval_losses = [log[f"eval_{loss}"] for log in trainer.state.log_history if f"eval_{loss}" in log]
        steps = list(range(1, len(train_losses) + 1))
        
        plt.figure(figsize=(10,5))
        plt.plot(steps, train_losses, label="Training Loss")
        plt.plot(steps, eval_losses, label="Validation Loss", linestyle="--")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss Curve ({loss})")
        plt.legend()
        
        plt.savefig(f"{args.exp_dir}/{loss}.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/train")
    parser.add_argument('--valid-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/test")
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--bins', default=None, help="for calculating accuracy-related metrics, it should be [1, 1.5, 2, 2.5, ...]")
    parser.add_argument('--exp-dir', type=str, default="exp-finetune/facebook/wav2vec2-large-xlsr-53")
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    # set seed
    print("[INFO] Set manual seed {}".format(args.seed))
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.enabled = False
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    #torch.use_deterministic_algorithms(True)
    
    main(args)
    
