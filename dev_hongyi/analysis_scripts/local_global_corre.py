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

    local_global_mean = scores_all[:, 0]
    local_global_std = scores_all[:, 1]
    iou_oracle = scores_all[:, 2]

    labels = (iou_oracle > 0.5).astype(float)

    pos = scores_all[labels > 0]
    neg = scores_all[labels == 0]
    # n_neg_max = int(min(len(neg), len(pos) * 2))
    # neg = neg[np.random.permutation(len(neg))[:n_neg_max]]

    plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(neg[:, 2], neg[:, 0], label="negative", s=2)
    plt.scatter(pos[:, 2], pos[:, 0], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("Ground-truth IoU")
    plt.ylabel("Local-global-sim-mean")

    plt.subplot(1, 2, 2)
    plt.scatter(neg[:, 2], neg[:, 1], label="negative", s=2)
    plt.scatter(pos[:, 2], pos[:, 1], label="positive", s=2)
    plt.grid()
    plt.legend()
    plt.xlabel("Ground-truth IoU")
    plt.ylabel("Local-global-sim-std")

    plt.tight_layout()
    plt.savefig("/home/s2139448/local_global_corre.png")