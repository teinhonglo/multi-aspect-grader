import argparse
import random
import logging
import os
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_dir",
                    default="../corpus/speaking/teemi_pretest/tb1",
                    type=str)

parser.add_argument("--data_dir",
                    default="data-speaking/teemi_pretest/tb1_qt1",
                    type=str)

parser.add_argument("--anno_fn",
                    default="annotation_multi_en_mct_cnn_tdnnf_tgt3meg-dl.xlsx",
                    type=str)

parser.add_argument("--id_column",
                    default="text_id",
                    type=str)

parser.add_argument("--text_column",
                    default="trans_stt",
                    type=str)

parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--sheet_name",
                    default="1", # "基礎聽答": 1, "情境式提問與問答": 2, "主題式口說任務": 3, "摘要報告": 4
                    type=str)

parser.add_argument("--all_bins",
                    default="1,1.5,2,2.5,3,3.5,4,4.5,5", # "基礎聽答": 1, "情境式提問與問答": 2, "主題式口說任務": 3, "摘要報告": 4
                    type=str)

parser.add_argument("--kfold",
                    default=5,
                    type=int)

parser.add_argument("--do_dig",
                    action="store_true")

parser.add_argument("--do_split",
                    action="store_true")

parser.add_argument("--test_on_valid",
                    action="store_true")

args = parser.parse_args()

corpus_dir = args.corpus_dir
data_dir = args.data_dir
anno_path = os.path.join(corpus_dir, args.anno_fn)
scores = args.scores.split()
id_column = args.id_column
text_column = args.text_column
kfold = args.kfold
sheet_name = args.sheet_name

def do_split(my_df, scores):
    new_dict = { k: [] for k in list(my_df.keys())}
    fk = list(my_df.keys())[0]
    for i in range(len(my_df[fk])):
        text_id = my_df["text_id"][i]
        texts = my_df["text"][i].split(" | ")
        wav_paths = my_df["wav_path"][i].split(" | ")

        for j in range(len(texts)):
            new_text_id = text_id + "_" + str(j)
            text = " ".join(texts[j].split())
            wav_path = " ".join(wav_paths[j].split())
            
            if text == "":
                text = "<SIL>"
            
            new_dict["text_id"].append(new_text_id)
            new_dict["text"].append(text)
            new_dict["wav_path"].append(wav_path)
            
            for score in scores:
                s = my_df[score][i]
                new_dict[score].append(s)

    return pd.DataFrame.from_dict(new_dict)


all_question_types = ["基礎聽答", "情境式提問與問答", "主題式口說任務", "摘要報告", "計分說明"]
question_dict = { qt: str(i+1) for i, qt in enumerate(all_question_types) }
inv_question_dict = { str(i+1): qt for i, qt in enumerate(all_question_types) }

xlsx_headers = ["text_id", "wav_path", "text" ] + scores
tsv_dict = {h:[] for h in xlsx_headers}
anno_df = pd.read_excel(anno_path, 
                        sheet_name=inv_question_dict[sheet_name], 
                        converters={id_column:str}
                        )

all_bins = np.array([float(ab) for ab in args.all_bins.split(",")])

for i, text_id in tqdm(enumerate(anno_df[id_column])):
     
    texts = anno_df[text_column][i]
    wav_paths = anno_df["wav_paths"][i]
    
    if not isinstance(text_id, str) or not isinstance(texts, str) or not isinstance(wav_paths, str):
        print(i)
        continue
    elif len(texts.split()) == 0:
        continue
    
    assert text_id not in tsv_dict["text_id"]
        
    tsv_dict["text_id"].append(text_id)
    tsv_dict["wav_path"].append(wav_paths)
    tsv_dict["text"].append(texts)
    
    for score in scores:
        anno_score = anno_df[score][i]
        #anno_score = (np.digitize(anno_score, cls_bins) + 1) / 2
        if args.do_dig:
            anno_score = np.digitize(anno_score, all_bins)
             
        tsv_dict[score].append(anno_score)
    
    before_text_id = text_id


from sklearn.model_selection import KFold
kf = KFold(n_splits=kfold, random_state=66, shuffle=True)

tsv_df = pd.DataFrame.from_dict(tsv_dict)

if args.test_on_valid:
    all_train_df = tsv_df
else:
    all_train_df, test_df = train_test_split(tsv_df, test_size=0.2, random_state=66)

for i, (train_index, valid_index) in enumerate(kf.split(all_train_df)):
    kfold_dir = str(i+1)
    result_dir = os.path.join(data_dir, kfold_dir)
    
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    
    train_df, valid_df = all_train_df.iloc[train_index], all_train_df.iloc[valid_index]
     
    if args.test_on_valid:
        test_df = valid_df
    
    train_df = train_df[train_df[scores[0]] != 0].reset_index(drop=True)
    valid_df = valid_df[valid_df[scores[0]] != 0].reset_index(drop=True)
    test_df = test_df[test_df[scores[0]] != 0].reset_index(drop=True)

    if args.do_split:
        train_df = do_split(train_df, scores)
        valid_df = do_split(valid_df, scores)
        test_df = do_split(test_df, scores)
    
    train_df.to_csv(os.path.join(result_dir, "train.tsv"), header=xlsx_headers, sep="\t", index=False)
    valid_df.to_csv(os.path.join(result_dir, "valid.tsv"), header=xlsx_headers, sep="\t", index=False) 
    test_df.to_csv(os.path.join(result_dir, "test.tsv"), header=xlsx_headers, sep="\t", index=False)

