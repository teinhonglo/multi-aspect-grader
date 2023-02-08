'''
Produce final report by averaging folds results.txt
also write all predictions for each scores

take
    score/fold/predictions.txt
write
    score/fold/results.txt (each fold results)
    score/predictions.txt (all predictions in kfolds)
    score/report.log (avg kfolds)
'''

import argparse
import numpy as np
from collections import defaultdict
from metrics_np import compute_metrics

parser = argparse.ArgumentParser()

parser.add_argument("--result_root",
                    default="runs/bert-model-writing",
                    type=str)

parser.add_argument("--bins",
                    type=str, 
                    help="for acc metrics when regression, it should be '0,0.5,1,1.5,2'")

parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--folds",
                    default="1 2 3 4 5",
                    type=str)

parser.add_argument("--merge-speaker",
                    action="store_true")

args = parser.parse_args()

result_root = args.result_root
bins = np.array([float(b) for b in args.bins.split(",")]) if args.bins else None
scores = args.scores.split()
folds = args.folds.split()
suffix = ".spk" if args.merge_speaker else ""

def predictions_to_list(predictions_file, merge_speaker=False, bins=""):

    # make dictionary
    pred_dict = defaultdict(list)
    label_dict = defaultdict(list)
    with open(predictions_file, "r") as rf:
        for line in rf.readlines():
            result = line.strip().split(" ")
            id, pred, label = result[0], result[1], result[2]
            if merge_speaker:
                # A01_u49_t9_p4_i19_1-5_20220919_0
                id = id.split('_')[1]
            pred_dict[id].append(float(pred))
            label_dict[id].append(float(label))

    ids, preds, labels = [], [], []
    for id, pred in pred_dict.items():
        pred = sum(pred) / len(pred)
        if not bins:
            pred = np.round(pred)
        label = sum(label_dict[id]) / len(label_dict[id])
        ids.append(id)
        preds.append(pred)
        labels.append(label)

    return ids, preds, labels

# NOTE: calculate each fold results then average
avg_results = {}
for score in scores:

    # calculate average results of k-folds
    avg_results[score] = defaultdict(lambda:0.0)

    # store all ids, preds, labels
    all_ids, all_preds, all_labels = [], [], []

    # for each fold calculate results
    for fold in folds:

        predictions_file = result_root + '/' + score + '/' + fold + '/predictions.txt'
        results_file = result_root + '/' + score + '/' + fold + '/results{}.txt'.format(suffix)
        total_losses = {}

        # get list of ids, preds, labels
        ids, preds, labels = predictions_to_list(predictions_file, merge_speaker=args.merge_speaker, bins=args.bins)
        compute_metrics(total_losses, np.array(preds), np.array(labels), bins=args.bins)

        # write fold results.txt
        with open(results_file, "w") as wf:

            if args.bins:
                print("fold {}, with bins {}\n".format(fold, bins))
                wf.write("with bins {}\n".format(bins))
            else:
                print("fold {}, without bins.\n".format(fold))
                wf.write("without bins.\n")

            for metrics, value in total_losses.items():
                print("{}: {}".format(metrics, value))
                wf.write("{}: {}\n".format(metrics, value))

                # avg results
                avg_results[score][metrics] += 1/len(folds) * float(value)

            print("\n")

        # store all ids, preds, labels
        all_ids += ids
        all_preds += preds
        all_labels += labels

    # NOTE: write all predictions for each score
    path = result_root + '/' + score + '/predictions{}.txt'.format(suffix)
    with open(path, "w") as wf:
        for id, pred, label in zip(all_ids, all_preds, all_labels):
            wf.write("{} {} {} \n".format(id, pred, label))

# NOTE: write report
path = result_root + '/report{}.log'.format(suffix)

with open(path, "w") as wf:
    wf.write("with bins {}\n".format(bins))
    wf.write("\n")
    for score in scores:
        wf.write("score: {}\n".format(score))
        for metrics, value in avg_results[score].items():
            wf.write("{}: {}\n".format(metrics, value))
        wf.write("\n")

'''
# calculate avg metrics
for score in scores:
    avg_results[score] = defaultdict(lambda:0.0)
    for fold in folds:
        path = result_root + '/' + score + '/' + fold
        with open(path + "/results.txt", "r") as rf:
            for i, line in enumerate(rf.readlines()):
                # bins info
                if i == 0:
                    bins = line
                    continue
                result = line.strip().split(": ")
                metrics, value = result[0], result[1]
                avg_results[score][metrics] += 1/len(folds) * float(value)

# write report
path = result_root + '/report.log'
with open(path, "w") as wf:
    wf.write(bins)
    wf.write("\n")
    for score in scores:
        wf.write("score: {}\n".format(score))
        for metrics, value in avg_results[score].items():
            wf.write("{}: {}\n".format(metrics, value))
        wf.write("\n")
'''
