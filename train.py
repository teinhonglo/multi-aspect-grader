#!/bin/python
import os
import sys
import json
import glob
import argparse
import torch
import torchaudio
from transformers import AutoConfig, AutoFeatureExtractor
from transformers import TrainingArguments, Trainer
from datasets import Dataset, Audio, load_from_disk

import numpy as np

# local import
from utils import make_dataset, DataCollatorCTCWithPadding, cal_class_weight, load_from_json, save_to_json
from metrics_np import compute_metrics
from model import AutoGraderModel, AutoGraderPrototypeModel

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

    # load wav2vec2 config
    config = AutoConfig.from_pretrained(
        model_args["model_path"],
        num_labels=model_args["num_labels"],
        problem_type=model_args["problem_type"],
        final_dropout=model_args["final_dropout"],
        gradient_checkpointing=True
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args["model_path"])

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
        print("[INFO] {} exists, using it".format(train_dataset_path + "/dataset.arrow"))
        tr_dataset = load_from_disk(train_dataset_path)

    # valid set
    valid_dataset_path = os.path.dirname(args.valid_json) + "/valid_dataset"
    if not os.path.exists(valid_dataset_path + "/dataset.arrow"):
        cv_dataset = make_dataset(args.valid_json)
        cv_dataset = cv_dataset.map(preprocess_function, num_proc=args.nj)
        cv_dataset.save_to_disk(valid_dataset_path)
    else:
        print("[INFO] {} exists, using it".format(valid_dataset_path + "/dataset.arrow"))
        cv_dataset = load_from_disk(valid_dataset_path)

    # data collator
    data_collator = DataCollatorCTCWithPadding(
        feature_extractor=feature_extractor, problem_type=model_args["problem_type"], padding=True
    )

    # NOTE: class_weight cal from trainset
    if model_args["class_weight_alpha"] != 0:
        assert model_args["problem_type"] == "single_label_classification"
        print("[INFO] Use class weight alpha {} ...".format(model_args["class_weight_alpha"]))
        class_weight = cal_class_weight(tr_dataset['labels'], model_args["num_labels"], \
            alpha=model_args["class_weight_alpha"]).to(device)
    else:
        print("[INFO] No class weight is provide ...")
        class_weight = None

    # NOTE: define model
    if "local_path" in model_args:
        print("[INFO] Load pretrained {} model from {} ...".format(model_args["model_type"], model_args["local_path"],))
        local_model = AutoGraderModel.from_pretrained(model_args["local_path"])
        if model_args["model_type"] == "prototype":
            model = AutoGraderPrototypeModel(model_args, class_weight=class_weight, pretrained=True)
        else:
            model = AutoGraderModel(model_args, class_weight=class_weight, pretrained=True)
        model.load_pretrained_wav2vec2(local_model.wav2vec2.state_dict())
    else:
        print("[INFO] Train a {} model from {} ...".format(model_args["model_type"], model_args["model_path"]))
        if model_args["model_type"] == "prototype":
            model = AutoGraderPrototypeModel(model_args, class_weight=class_weight, pretrained=True)
            if model_args["init_prototypes"]:
                model.init_prototypes(tr_dataset, path=train_dataset_path)
        else:
            model = AutoGraderModel(model_args, class_weight=class_weight, pretrained=True)
        model.freeze_feature_extractor()
    torch.save(model.config, args.exp_dir + '/config.pth')
    #torch.save(model.model, args.exp_dir + '/encoder_model.pth')

    # NOTE: define metric
    def calculate_metrics(pred):

        # preds (0-7 to 1-8)
        # NOTE: teemi: 0-8 to 1-9
        preds = pred.predictions
        preds = np.argmax(preds, axis=1) + 1 \
            if model_args["problem_type"] == "single_label_classification" else preds
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/train")
    parser.add_argument('--valid-json', type=str, help="kaldi-format data", default="/share/nas167/fuann/asr/gop_speechocean762/s5/data/test")
    parser.add_argument('--train-conf', type=str)
    parser.add_argument('--bins', default=None, help="for calculating accuracy-related metrics, it should be [1, 1.5, 2, 2.5, ...]")
    parser.add_argument('--exp-dir', type=str, default="exp-finetune/facebook/wav2vec2-large-xlsr-53")
    parser.add_argument('--nj', type=int, default=4)
    args = parser.parse_args()

    main(args)
