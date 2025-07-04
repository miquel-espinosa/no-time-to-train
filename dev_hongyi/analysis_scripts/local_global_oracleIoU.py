import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == "__main__":
    with open("scalars_all.pkl", "rb") as f:
        data = pickle.load(f)

    scores_all = np.concatenate(data, axis=0)
    n_data = float(len(scores_all))
    print("Data size:", n_data)

    local_sims = scores_all[:, 0]
    global_sims = scores_all[:, 1]
    iou_oracle = scores_all[:, 2]

    labels = (iou_oracle > 0.5).astype(float)

    pos = scores_all[labels > 0]
    neg = scores_all[labels == 0]

    # n_neg_max = int(min(len(neg), len(pos) * 2))
    # neg = neg[np.random.permutation(len(neg))[:n_neg_max]]

    plt.figure(figsize=(11, 11))

    plt.subplot(2, 2, 1)
    plt.scatter(neg[:, 0], neg[:, 2], label="negative", s=2)
    plt.scatter(pos[:, 0], pos[:, 2], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("Local Similarity")
    plt.ylabel("Ground-truth IoU")

    plt.subplot(2, 2, 2)
    plt.scatter(neg[:, 0], neg[:, 1], label="negative", s=2)
    plt.scatter(pos[:, 0], pos[:, 1], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("Local Similarity")
    plt.ylabel("Global Similarity")

    plt.subplot(2, 2, 3)
    plt.scatter(neg[:, 1], neg[:, 2], label="negative", s=2)
    plt.scatter(pos[:, 1], pos[:, 2], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("Global Similarity")
    plt.ylabel("Ground-truth IoU")

    plt.subplot(2, 2, 4)
    plt.scatter(neg[:, 1] * neg[:, 0], neg[:, 2], label="negative", s=2)
    plt.scatter(pos[:, 1] * pos[:, 0], pos[:, 2], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("Local * Global")
    plt.ylabel("Ground-truth IoU")

    plt.tight_layout()
    plt.savefig("/home/s2139448/local_global_analysis.png")


    print("Local similarity analysis:")
    intervals = np.linspace(local_sims.min(), local_sims.max(), 20)
    for s in intervals:
        pred_labels = (local_sims >= s).astype(float)

        tp = (labels == pred_labels)[labels == 1].sum()
        fp = (labels != pred_labels)[labels == 1].sum()

        acc = (labels == pred_labels).sum() / n_data
        recall = (labels == pred_labels)[labels == 1].sum() / labels.sum()
        precision = (labels == pred_labels)[labels == 1].sum() / pred_labels.sum()
        f1 = 1. / (1. / recall + 1. / precision)

        print("Thr: %.4f, Acc: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f" % (s, acc, recall, precision, f1))

    print("AUC analysis")
    exps = np.linspace(0.0, 1.0, 21)
    for exp in exps:
        merged_scores = global_sims * local_sims**exp
        n_intervals = 100
        intervals = np.linspace(merged_scores.min(), merged_scores.max(), n_intervals)
        recalls, precisions = [], []
        for s in intervals:
            pred_labels = (merged_scores >= s).astype(float)

            tp = (labels == pred_labels)[labels == 1].sum()
            fp = (labels != pred_labels)[labels == 1].sum()

            acc = (labels == pred_labels).sum() / n_data
            recall = (labels == pred_labels)[labels == 1].sum() / labels.sum()
            precision = (labels == pred_labels)[labels == 1].sum() / pred_labels.sum()
            recalls.append(recall)
            precisions.append(precision)

        recalls.reverse()
        precisions.reverse()
        auc = 0.0
        for i in range(n_intervals-1):
            auc += 0.5 * (precisions[i] + precisions[i+1]) * (recalls[i+1] - recalls[i])
        print("Exp: %.2f, AUC: %.4f" % (exp, auc))

