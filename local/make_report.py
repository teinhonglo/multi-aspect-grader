'''
Produce final report by averaging folds results.txt
'''

import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--result_root",
                    default="runs/bert-model-writing",
                    type=str)

parser.add_argument("--scores",
                    default="content pronunciation vocabulary",
                    type=str)

parser.add_argument("--folds",
                    default="1 2 3 4 5",
                    type=str)

args = parser.parse_args()

result_root = args.result_root
scores = args.scores.split()
folds = args.folds.split()

avg_results = {}

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

