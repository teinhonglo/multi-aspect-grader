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
    "label":[]
}

with open(args.tsv, "r") as rf:
    # text_id, wav_path, text, holistic
    for i, line in enumerate(rf.readlines()):
        
        if i == 0: 
            columns = {key:header_index for header_index, key in enumerate(line.strip().split('\t'))}
            continue

        temp = line.strip().split('\t')

        wav = temp[columns['wav_path']]
        text = temp[columns['text']]
        label = float(temp[columns[args.score]])
        
        text_id = temp[columns['text_id']]
        
        data_dict["id"].append(text_id)
        data_dict["audio"].append(wav)
        data_dict["text"].append(text)
        data_dict["label"].append(label)


with open(args.json, 'w') as jsonfile:
    json.dump(data_dict, jsonfile)
