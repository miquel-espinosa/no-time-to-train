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
    mem_ins_sim = scores_all[:, 3]


    min_pos_scores = []
    mem_ins_sims = []
    for i in range(80):
        scoores_cls = scores_all[categories==i]
        iou_cls = iou_oracle[categories==i]

        if scoores_cls.shape[0] == 0:
            continue

        mem_ins_sim_cls = mem_ins_sim[categories==i][0]

        labels = (iou_cls > 0.5).astype(float)

        pos = scoores_cls[labels > 0]
        neg = scoores_cls[labels == 0]

        if len(pos) > 0:
            # k = np.log(pos[:, 0].min()) / np.log(mem_ins_sim_cls)
            # k = pos[:, 0].min() / mem_ins_sim_cls
            print(str(classes[i])+':', pos[:, 0].min(), mem_ins_sim_cls)

            min_pos_scores.append(pos[:, 0].min())
            mem_ins_sims.append(mem_ins_sim_cls)

        plt.scatter(neg[:, 0], neg[:, 2], label="negative", s=5)
        plt.scatter(pos[:, 0], pos[:, 2], label="positive", s=5)
        plt.plot([mem_ins_sim_cls, mem_ins_sim_cls], [-0.1, 1.0], color='grey')
        plt.grid()
        plt.legend()
        plt.xlim(0., 1.)
        plt.ylim(-0.1, 1.)
        plt.xlabel("Global Similarity")
        plt.ylabel("Ground-truth IoU")
        plt.title(str(classes[i]))

        plt.tight_layout()
        plt.savefig("./result_analysis/figures/avgSim_%d_%s.png"%(i, str(classes[i])))
        plt.close()

    plt.scatter(min_pos_scores, mem_ins_sims, s=5)
    plt.plot([0, 1], [0, 1], color='grey')
    plt.savefig("./result_analysis/figures/minPosScore_memInsSims.png")