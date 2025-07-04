import pickle
import random
import json
from collections import OrderedDict

from pycocotools.coco import COCO

import numpy as np

def sample_small_dataset(json_file, output_path, size=100):
    with open(json_file) as f:
        data = json.load(f)

    img_ids = [x["id"] for x in data["images"]]
    random.shuffle(img_ids)
    sampled_img_ids = img_ids[:size]
    tmp_dict = dict()
    for img_id in sampled_img_ids:
        tmp_dict[img_id] = 0

    sampled_images = []
    sampled_annotations = []
    categories = data["categories"]

    for img in data["images"]:
        if img["id"] in tmp_dict.keys():
            sampled_images.append(img)
    for ann in data["annotations"]:
        if ann["image_id"] in tmp_dict.keys():
            sampled_annotations.append(ann)

    output = dict(
        categories=categories,
        images=sampled_images,
        annotations=sampled_annotations
    )

    with open(output_path, 'w') as fw:
        json.dump(output, fw)


if __name__ == "__main__":
    sample_small_dataset(
        f"./data/coco/annotations/instances_val2017.json",
        f"./data/coco/annotations_refsam2/val2017_500.json",
        size=500
    )