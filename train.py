#!/bin/python
import os
import json
import argparse
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from transformers import TrainingArguments, Trainer
from datasets import Dataset, Audio, load_from_disk

import numpy as np

# local import 
from utils import make_dataset, DataCollatorCTCWithPadding, cal_class_weight, load_from_json
from metrics_np import compute_metrics
from model import Wav2vec2GraderModel

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(
        args.model_path, 
        num_labels=args.num_labels, 
        problem_type=args.problem_type
        
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_path)

    # data preprocess
    def preprocess_function(batch):
        audio = batch["audio"]
        # extract features return input_values
        batch["input_values"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["labels"] = batch['label']
        return batch

    # train set
    train_dataset_path = os.path.dirname(args.train_json) + "/train_dataset"
    if not os.path.exists(train_dataset_path + "/dataset.arrow"):
        tr_dataset = make_dataset(args.train_json)
        tr_dataset = tr_dataset.map(preprocess_function, num_proc=args.nj)
        tr_dataset.save_to_disk(train_dataset_path)
    else:
        tr_dataset = load_from_disk(train_dataset_path)
    
    # valid set
    valid_dataset_path = os.path.dirname(args.valid_json) + "/valid_dataset"
    if not os.path.exists(valid_dataset_path + "/dataset.arrow"):
        cv_dataset = make_dataset(args.valid_json)
        cv_dataset = cv_dataset.map(preprocess_function, num_proc=args.nj)
        cv_dataset.save_to_disk(valid_dataset_path)
    else:
        cv_dataset = load_from_disk(valid_dataset_path)

    # data collator
    data_collator = DataCollatorCTCWithPadding(
        feature_extractor=feature_extractor, problem_type=args.problem_type, padding=True
    )

    # save to exp_dir
    #config.save_pretrained(args.exp_dir)
    #feature_extractor.save_pretrained(args.exp_dir)
    #print("Save config/extractor to {} ...".format(args.exp_dir))

    # NOTE: class_weight cal from trainset
    if args.class_weight_alpha != 0:
        assert args.problem_type == "single_label_classification"
        print("[INFO] Use class weight alpha {} ...".format(args.class_weight_alpha))
        class_weight = cal_class_weight(tr_dataset['label'], args.num_labels, \
            alpha=args.class_weight_alpha).to(device)
        #print(class_weight)
        #input()
    else:
        print("[INFO] No class weight is provide ...")
        class_weight = None
    
    # NOTE: define model
    model = Wav2vec2GraderModel.from_pretrained(args.model_path, config=config, class_weight=class_weight)
    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()

    # NOTE: define metric
    def calculate_metrics(pred):
        
        # preds (0-7 to 1-8)
        preds = pred.predictions
        preds = np.argmax(preds, axis=1) + 1 \
            if args.problem_type == "single_label_classification" else preds
        # labels
        labels = pred.label_ids

        print("\n\n")
        print("predictions:")
        print("{}".format(preds))
        print("labels:")
        print("{}".format(labels))
        print("\n\n")
        
        # metrics
        total_losses = {}
        compute_metrics(total_losses, np.array(preds), np.array(labels))
        return total_losses

    # NOTE: define training args
    train_args = load_from_json(args.train_conf)
    training_args = TrainingArguments(
        output_dir=args.exp_dir,
        group_by_length=True,
        fp16=True,
        gradient_checkpointing=True,
        metric_for_best_model="within_0.5",
        greater_is_better=True,
        load_best_model_at_end=True,
        **train_args
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=calculate_metrics,
        train_dataset=tr_dataset,
        eval_dataset=cv_dataset,
        tokenizer=feature_extractor,
    )

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # save the best model
    best_path = os.path.join(args.exp_dir, 'best')
    trainer.save_model(best_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/train")
    parser.add_argument('--valid-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/test")
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--problem-type', default="regression", choices=['regression', 'single_label_classification'])
    parser.add_argument('--class-weight-alpha', type=float, default=0)
    parser.add_argument('--num-labels', type=int, default=1)
    parser.add_argument('--model-path', type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument('--exp-dir', type=str, default="exp-finetune/facebook/wav2vec2-large-xlsr-53")
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    main(args)