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
from model import Wav2vec2GraderModel

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(args.model_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)
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

    # show metrics
    print("predictions:")
    print("{}".format(results["pred"]))
    print("labels:")
    print("{}".format(results["label"]))

    '''
    total_losses = {}
    compute_metrics(total_losses, np.array(results["pred"]), np.array(results["label"]), bins=args.bins)
    # show & write results
    results_file = os.path.join(args.exp_dir, "results.txt")
    with open(results_file, 'w') as wf:
        if args.bins:
            bins = np.array([float(b) for b in args.bins.split(",")]) if args.bins else None
            print("with bins {}\n".format(bins))
            wf.write("with bins {}\n".format(bins))
        else:
            print("without bins.\n")
            wf.write("without bins.\n")
        for metrics, value in total_losses.items():
            print("{}: {}".format(metrics, value))
            wf.write("{}: {}\n".format(metrics, value))
    '''

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
    parser.add_argument('--bins', type=str, help="for calculating accuracy-related metrics, it should be [0, 0.5, 1, 1.5, ...]")
    parser.add_argument('--exp-dir', type=str)
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    main(args)
