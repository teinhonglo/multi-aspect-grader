import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import json
import argparse


def plot(log_dir, outputs, prefix=""):
    pred_labels, gold_labels, outputs_mean = [], [], []
    
    gold_labels = np.array(outputs['gold_labels']) - 1
    pred_labels = np.array(outputs['pred_labels']) - 1
    outputs_mean = np.array(outputs['outputs_mean'])
     
    # 建立 TSNE 模型並降維
    # 注意: 只有ICNALE是對的
    #if len(unique_levels) != 5:
    #    return
    levels = ["A2", "B1_1", "B1_2", "B2", "C"]
    colors = ['red', 'green', 'blue', 'orange', 'purple']
             
    #pca = PCA(n_components=50)
    #X_pca = pca.fit_transform(outputs_mean)
    X_pca = outputs_mean
            
    model = TSNE(n_components=2, init="pca", learning_rate="auto", verbose=1, perplexity=30, n_iter=5000)
    tsne_features = model.fit_transform(X_pca)
    plt.clf()

    for i in range(len(levels)):
        lv_idx = np.where(gold_labels == i)
        plt.scatter(tsne_features[lv_idx, 0], tsne_features[lv_idx, 1], c=colors[i], label=levels[i])
    
    #plt.legend(loc="upper right")
    plt.savefig(log_dir + '/' + prefix + 'tsne_plot_gold_labels.png', pad_inches=0)
    
    plt.clf()
    for i in range(len(levels)):
        lv_idx = np.where(pred_labels == i)
        plt.scatter(tsne_features[lv_idx, 0], tsne_features[lv_idx, 1], c=colors[i], label=levels[i])
    
    #plt.legend(loc="upper right")
    plt.savefig(log_dir + '/' + prefix + 'tsne_plot_pred_labels.png', pad_inches=0)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot TSNE')
    parser.add_argument('--result_root', help='experiment directory', type=str, default='exp/icnale/train_icnale_baseline_cls_wav2vec2/holistic/1')
    parser.add_argument('--is_proto', help='experiment directory', type=str, default='false')
    args = parser.parse_args()

    result_root = args.result_root
    
    if args.is_proto == "true":
        is_proto = True
    else:
        is_proto = False
    
    # test_set
    embeds_json_fn = os.path.join(result_root, "embeds.json")

    with open(embeds_json_fn, "r") as fn:
        embeds_json = json.load(fn)
     
    outputs = {"gold_labels": [], "pred_labels": [], "outputs_mean": []}
    
    for text_id in list(embeds_json.keys()):
        outputs["gold_labels"].append(embeds_json[text_id]["label"])
        outputs["pred_labels"].append(embeds_json[text_id]["pred"])
        outputs["outputs_mean"].append(embeds_json[text_id]["embed"][0])

    if is_proto:
        protos_json_fn = os.path.join(result_root, "protos.json")

        with open(protos_json_fn, "r") as fn:
            protos_json = json.load(fn)
        
        for proto_id in list(protos_json.keys()):
            labels, _ = proto_id.split("_")
            outputs["gold_labels"].append(labels)
            outputs["pred_labels"].append(labels)
            outputs["outputs_mean"].append(protos_json[proto_id]["embed"])


    plot(result_root, outputs)
