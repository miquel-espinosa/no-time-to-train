import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


classes = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

if __name__ == "__main__":
    with open("scalars_all.pkl", "rb") as f:
        data = pickle.load(f)

    scores_all = np.concatenate(data, axis=0)
    n_data = float(len(scores_all))

    sims = scores_all[:, 0]
    categories = scores_all[:, 1].astype(int)
    iou_oracle = scores_all[:, 2]

    for i in range(80):
        scores_cls = sims[categories==i]
        iou_cls = iou_oracle[categories==i]

        low_thr = 0.0
        inds = scores_cls > low_thr
        scores_cls = scores_cls[inds]
        iou_cls = iou_cls[inds]

        if scores_cls.shape[0] == 0:
            continue

        labels = (iou_cls > 0.5).astype(float)

        pos_scores = scores_cls[labels > 0].reshape(-1)
        neg_scores = scores_cls[labels == 0].reshape(-1)

        bins = np.linspace(low_thr, 1., 50)
        plt.hist(pos_scores, bins=bins, label="positive", density=False, alpha=0.5)
        plt.hist(neg_scores, bins=bins, label="negative", density=False, alpha=0.5)
        plt.grid()
        plt.legend()
        plt.xlim(0., 1.)
        plt.xlabel("Global Similarity")
        plt.ylabel("Density")
        plt.title(str(classes[i]))

        plt.tight_layout()
        plt.savefig("./result_analysis/figures/score_hist_%d_%s.png"%(i, str(classes[i])))
        plt.close()


    # All class
    inds = sims > low_thr
    scores_cls = sims[inds]
    iou_cls = iou_oracle[inds]

    labels = (iou_cls > 0.5).astype(float)
    pos_scores = scores_cls[labels > 0].reshape(-1)
    neg_scores = scores_cls[labels == 0].reshape(-1)

    bins = np.linspace(low_thr, 1., 50)
    plt.hist(pos_scores, bins=bins, label="positive", density=False, alpha=0.5)
    plt.hist(neg_scores, bins=bins, label="negative", density=False, alpha=0.5)
    plt.grid()
    plt.legend()
    plt.xlim(0., 1.)
    plt.xlabel("Global Similarity")
    plt.ylabel("Density")
    plt.title("All classes")

    plt.tight_layout()
    plt.savefig("./result_analysis/figures/score_hist_allClasses.png")

