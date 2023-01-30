#!/bin/python
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--score", type=str)
parser.add_argument("--tsv", type=str)
parser.add_argument("--json", type=str)
args = parser.parse_args()

data_dict = {
    "id":[], 
    "audio":[], 
    "text":[], 
    "label":[],
}

with open(args.tsv, "r") as rf:
    # text_id, wav_path, text, holistic
    for i, line in enumerate(rf.readlines()):
        
        if i == 0: 
            continue

        temp = line.strip().split('\t')

        wav = temp[1]
        text = temp[2]
        if args.score == "holistic":
            label = int(float(temp[3]))
        else:
            raise ValueError("No score {} is provided".format(args.score))

        basename = os.path.basename(wav).split('.')[0]
        data_dict["id"].append(basename)
        data_dict["audio"].append(wav)
        data_dict["text"].append(text)
        data_dict["label"].append(label)

with open(args.json, 'w') as jsonfile:
    json.dump(data_dict, jsonfile)