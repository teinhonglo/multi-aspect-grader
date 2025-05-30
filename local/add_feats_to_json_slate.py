import argparse
import random
import logging
import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pandas as pd
import json

'''
Added prompt and human_feats (delivery & language use) to exisiting json

cd /share/nas167/teinhonglo/AcousticModel/spoken_test/asr-whisper
python local/data/create_feats_from_json.py
'''

parser = argparse.ArgumentParser()

parser.add_argument("--src_data_dir",
                    default="data-json/icnale/trans_stt_whisper_large/holistic/1",
                    type=str)

parser.add_argument("--dst_data_dir",
                    default="data-json/icnale/trans_stt_whisper_large_multi_aspect/holistic/1",
                    type=str)

parser.add_argument("--json_files",
                    default="train.json,valid.json,test.json",
                    type=str)

parser.add_argument("--multi_aspect_json_file",
                    default="/share/nas167/teinhonglo/AcousticModel/spoken_test/asr-esp/data/icnale/icnale_monologue/whisperx_large-v1/aspect_feats.json",
                    type=str)                    

args = parser.parse_args()

src_data_dir = args.src_data_dir
dst_data_dir = args.dst_data_dir
json_files = args.json_files.split(",")
multi_aspect_json_file = args.multi_aspect_json_file

if not os.path.isdir(dst_data_dir):
    os.makedirs(dst_data_dir)

def add_prompt_info(json_dict, multi_aspect_dict):
    json_dict["prompt"] = []
    json_dict["delivery"] = []
    json_dict["language_use"] = []

    for uttid in json_dict["id"]:
        info = uttid.split("_")
        prompt = "SLATE"
        
        delivery = [] #multi_aspect_dict[uttid]["delivery"]
        language_use = [] #multi_aspect_dict[uttid]["language_use"]
        
        json_dict["prompt"].append(prompt)
        json_dict["delivery"].append(delivery)
        json_dict["language_use"].append(language_use)
    
    return json_dict


for json_file in json_files:
    with open(os.path.join(src_data_dir, json_file), "r") as fn:
        json_dict = json.load(fn)

    with open(multi_aspect_json_file, "r") as fn:
        multi_aspect_dict = json.load(fn)
    
    json_dict = add_prompt_info(json_dict, multi_aspect_dict)
    print(os.path.join(dst_data_dir, json_file))

    with open(os.path.join(dst_data_dir, json_file), "w") as fn:
        json.dump(json_dict, fn)
