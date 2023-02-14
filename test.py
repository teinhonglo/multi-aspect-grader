#!/bin/python
import os
import json
import argparse
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import TrainingArguments, Trainer
from datasets import Dataset, Audio, load_metric, load_from_disk

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# local import
from utils import make_dataset
#from metrics_np import compute_mse, _accuracy_within_margin
from metrics_np import compute_metrics
from model import Wav2vec2GraderModel, Wav2vec2GraderPrototypeModel

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
    if args.model_type == 'prototype':
        model = Wav2vec2GraderPrototypeModel.from_pretrained(args.model_path).to(device)
    else:
        model = Wav2vec2GraderModel.from_pretrained(args.model_path).to(device)

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
            logits = model(input_values, return_dict=True).logits

        if config.problem_type == "single_label_classification":
            pred_ids = torch.argmax(logits, dim=-1) + 1
        else:
            pred_ids = logits

        pred_ids = pred_ids.detach().cpu().numpy().item()
        batch["pred"] = pred_ids

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-json', type=str)
    parser.add_argument('--model-path', type=str, default="facebook/wav2vec2-base")
    parser.add_argument('--model-type', default="baseline", choices=['baseline', 'prototype'])
    parser.add_argument('--bins', type=str, help="for calculating accuracy-related metrics, it should be [0, 0.5, 1, 1.5, ...]")
    parser.add_argument('--exp-dir', type=str)
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    main(args)
