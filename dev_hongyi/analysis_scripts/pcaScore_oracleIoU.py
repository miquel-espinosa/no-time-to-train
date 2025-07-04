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

    pca_scores = scores_all[:, 0]
    similarities = scores_all[:, 1]
    iou_oracle = scores_all[:, 2]

    labels = (iou_oracle > 0.5).astype(float)

    pos = scores_all[labels > 0]
    neg = scores_all[labels == 0]
    n_neg_max = int(min(len(neg), len(pos) * 2))
    neg = neg[np.random.permutation(len(neg))[:n_neg_max]]

    plt.figure(figsize=(11, 11))

    plt.subplot(2, 2, 1)
    plt.scatter(neg[:, 0], neg[:, 2], label="negative", s=2)
    plt.scatter(pos[:, 0], pos[:, 2], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("PCA Score")
    plt.ylabel("Ground-truth IoU")

    plt.subplot(2, 2, 2)
    plt.scatter(neg[:, 0], neg[:, 1], label="negative", s=2)
    plt.scatter(pos[:, 0], pos[:, 1], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("PCA Score")
    plt.ylabel("DINO v2 Similarity")

    plt.subplot(2, 2, 3)
    plt.scatter(neg[:, 1], neg[:, 2], label="negative", s=2)
    plt.scatter(pos[:, 1], pos[:, 2], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("DINO v2 Similarity")
    plt.ylabel("Ground-truth IoU")

    plt.subplot(2, 2, 4)
    plt.scatter(neg[:, 1] * neg[:, 0], neg[:, 2], label="negative", s=2)
    plt.scatter(pos[:, 1] * pos[:, 0], pos[:, 2], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("Similarity * PCA")
    plt.ylabel("Ground-truth IoU")

    plt.tight_layout()
    plt.savefig("/home/s2139448/pca_analysis.png")


    print("Merged score analysis:")
    merge_score = pca_scores * similarities
    pca_intervals = np.linspace(merge_score.min(), merge_score.max(), 20)
    for s in pca_intervals:
        pred_labels = (pca_scores >= s).astype(float)

        tp = (labels == pred_labels)[labels == 1].sum()
        fp = (labels != pred_labels)[labels == 1].sum()

        acc = (labels == pred_labels).sum() / n_data
        recall = (labels == pred_labels)[labels == 1].sum() / labels.sum()
        precision = (labels == pred_labels)[labels == 1].sum() / pred_labels.sum()
        f1 = 1. / (1. / recall + 1. / precision)

        print("Thr: %.4f, Acc: %.4f, Recall: %.4f, Precision: %.4f, F1: %.4f" % (s, acc, recall, precision, f1))








