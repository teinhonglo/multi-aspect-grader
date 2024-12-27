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

prompt_info_fn = "/share/corpus/2023_teemiv2/prompts.json"
with open(prompt_info_fn, "r") as fn:
    prompt_info = json.load(fn)

def add_prompt_info(json_dict, multi_aspect_dict):
    json_dict["prompt"] = []
    json_dict["delivery"] = []
    json_dict["language_use"] = []

    for uttid in json_dict["id"]:
        # B08_u3024_t68_p25_i61_2-1_20231012
        info = uttid.split("-")
        # A08_01
        item_id = info[0].split("_")[0]
        sub_item_id = info[1].split("_")[0]
        prompt_id = f"{item_id}_0{sub_item_id}"
        prompt = []

        if prompt_id in prompt_info:
            if "description" in prompt_info:
                prompt.append(prompt_info[prompt_id]["description"])
            
            if "question" in prompt_info:
                prompt.append(prompt_info[prompt_id]["question"])
            
            prompt = " ".join(prompt)    
        else:
            raise ValueError(prompt_id)
        
        delivery = multi_aspect_dict[uttid]["delivery"]
        language_use = multi_aspect_dict[uttid]["language_use"]
        
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
