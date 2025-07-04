import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == "__main__":
    with open("triplets_all.pkl", "rb") as f:
        data = pickle.load(f)

    triplets = np.concatenate(data, axis=0)
    n_data = float(len(triplets))
    print("Data size:", len(triplets))

    similarities = triplets[:, 0]
    pred_ious = triplets[:, 1]
    scores_oracle = triplets[:, 2]

    labels = (scores_oracle > 0.5).astype(float)

    pos = triplets[labels > 0]
    neg = triplets[labels == 0]
    neg = neg[np.random.permutation(len(neg))[:len(pos)]]
    plt.scatter(pos[:, 0], pos[:, 1], label="positive", s=2)
    plt.scatter(neg[:, 0], neg[:, 1], label="negative", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("DINO v2 similarity")
    plt.ylabel("Predicted IoU")
    plt.savefig("/home/s2139448/sam2_matching_analysis.png")


    sim_intervals = np.linspace(similarities.min(), similarities.max(), 20)
    print("Similarity analysis:")
    for s in sim_intervals:
        pred_labels = (similarities >= s).astype(float)

        tp = (labels == pred_labels)[labels == 1].sum()
        fp = (labels != pred_labels)[labels == 1].sum()

        acc = (labels == pred_labels).sum() / n_data
        recall = (labels == pred_labels)[labels == 1].sum() / labels.sum()
        precision = (labels == pred_labels)[labels == 1].sum() / pred_labels.sum()
        f1 = 1. / (1./recall + 1./precision)

        print("Thr: %.4f, Acc: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f"%(s, acc, recall, precision, f1))

    print("IoU analysis:")
    iou_intervals = np.linspace(0, 1.0, 20)
    for s in iou_intervals:
        pred_labels = (pred_ious >= s).astype(float)

        tp = (labels == pred_labels)[labels == 1].sum()
        fp = (labels != pred_labels)[labels == 1].sum()

        acc = (labels == pred_labels).sum() / n_data
        recall = (labels == pred_labels)[labels == 1].sum() / labels.sum()
        precision = (labels == pred_labels)[labels == 1].sum() / pred_labels.sum()
        f1 = 1. / (1. / recall + 1. / precision)

        print("Thr: %.4f, Acc: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f"%(s, acc, recall, precision, f1))

    print("Designed metric")
    alphas = np.linspace(0.0, 1.0, 11)
    for a in alphas:
        scores = similarities**a * pred_ious**(1-a)
        interval = np.linspace(scores.min(), scores.max(), 20)
        print("Alpha: %.4f"%a)
        for s in interval:
            pred_labels = (scores >= s).astype(float)

            tp = (labels == pred_labels)[labels == 1].sum()
            fp = (labels != pred_labels)[labels == 1].sum()

            acc = (labels == pred_labels).sum() / n_data
            recall = (labels == pred_labels)[labels == 1].sum() / labels.sum()
            precision = (labels == pred_labels)[labels == 1].sum() / pred_labels.sum()
            f1 = 1. / (1. / recall + 1. / precision)

            print("Thr: %.4f, Acc: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f" % (s, acc, recall, precision, f1))