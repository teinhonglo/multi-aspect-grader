'''
take predictions.txt, plot confusion matrix.
'''

import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()

parser.add_argument("--result_root",
                    default="runs-speaking/gept-p3/trans_stt_tov_round/bert-model",
                    type=str)

parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--folds",
                    default="1 2 3 4 5",
                    type=str)

parser.add_argument("--bins",
                    default="1,2,2.5,3,3.5,4,4.5,5",
                    type=str)

parser.add_argument("--labels",
                    default="pre-A,A1,A1A2,A2,A2B1,B1,B1B2,B2",
                    type=str)

parser.add_argument("--merge-speaker",
                    action="store_true")

args = parser.parse_args()

result_root = args.result_root
scores = args.scores.split()
folds = args.folds.split()
bins = np.array([ float(ab) for ab in args.bins.split(",")]) if args.bins else None
labels = args.labels.split(",")
suffix = ".spk" if args.merge_speaker else ""

for score in scores:

    all_preds = []
    all_labels = []

    # take all preds, labels in predictions.txt
    path = result_root + '/' + score
    with open(path + "/predictions{}.txt".format(suffix), "r") as rf:
        for line in rf.readlines():
            result = line.strip().split(" ")
            id, pred, label = result[0], result[1], result[2]
            all_preds.append(float(pred))
            all_labels.append(float(label))

    # cal confusion matrix
    file_name = os.path.join(result_root, score)
    png_name = file_name + "{}.png".format(suffix)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if args.bins:
        all_preds = np.digitize(all_preds, bins)
        all_labels = np.digitize(all_labels, bins)
    else:   # for classification, map 1 ~ n_labels to 0 ~ n_labels-1
        all_preds = all_preds - 1
        all_labels = all_labels - 1

    conf_mat = confusion_matrix(all_labels, all_preds, labels=range(len(labels)))
    row_sum = np.sum(conf_mat, axis = 1)
    conf_mat_prec = conf_mat / row_sum[:, np.newaxis]
    conf_mat_prec[np.where(conf_mat == 0)] = 0
    conf_mat_df = pd.DataFrame(conf_mat, index=labels, columns=labels)
    conf_mat_prec_df = pd.DataFrame(conf_mat_prec, index=labels, columns=labels)

    # save heatmap
    # cbar # bar
    sns.heatmap(data=conf_mat_prec_df, annot=conf_mat_df, fmt='g', cbar=False)
    plt.xlabel("Predictions")
    plt.ylabel("Annotations")
    plt.savefig(png_name, pad_inches=0)
    plt.clf()
