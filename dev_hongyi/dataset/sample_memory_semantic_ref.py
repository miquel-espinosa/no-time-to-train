import pickle
import random
import json
from collections import OrderedDict

from pycocotools.coco import COCO

METAINFO = {
    'default_classes':
        (
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
}



def sample_semantic_ref_memory_dataset(json_file, out_path, memory_length, cat_names=[]):
    coco = COCO(json_file)
    cat_ids = coco.getCatIds(catNms=METAINFO['default_classes'])

    cat_to_imgs = {}
    img_to_anns = {}
    for ann_id, ann in coco.anns.items():
        if ann["category_id"] not in cat_ids:
            continue
        ann_cat_id = ann["category_id"]

        if ann_cat_id not in cat_to_imgs.keys():
            cat_to_imgs[ann_cat_id] = []
        if ann['image_id'] not in cat_to_imgs[ann_cat_id]:
            cat_to_imgs[ann_cat_id].append(ann["image_id"])

        if ann['image_id'] not in img_to_anns.keys():
            img_to_anns[ann['image_id']] = []
        img_to_anns[ann['image_id']].append(ann_id)

    for cat_id in cat_to_imgs.keys():
        random.shuffle(cat_to_imgs[cat_id])

    sampled_data_by_cat = OrderedDict()
    for cat_id in cat_to_imgs.keys():
        sampled_data_by_cat[cat_id] = []
        if len(cat_to_imgs[cat_id]) < memory_length:
            raise ValueError("Reference for class %d is not enough" % cat_id)
        for img_id in cat_to_imgs[cat_id][:memory_length]:
            sampled_data_by_cat[cat_id].append({"img_id": img_id, "ann_ids": []})
            for ann_id in img_to_anns[img_id]:
                if coco.loadAnns([ann_id])[0]['category_id'] == cat_id:
                    sampled_data_by_cat[cat_id][-1]["ann_ids"].append(ann_id)

    with open(out_path, 'wb') as fw:
        pickle.dump(sampled_data_by_cat, fw)
    print("Results output to: %s" % out_path)




if __name__ == "__main__":

    user = 'hongyi'  # 'miguel'

    if user == 'hongyi':
        base_path = '/home/s2139448/projects/sam2_ref'
    elif user == 'miguel':
        base_path = '/home/s2254242/projects/finetune-SAM2'

    annotations_path = f"{base_path}/data/coco/annotations_refsam2"

    memory_length = 4  # Needs to be the same (or lower) than the trained checkpoint

    all_refs_json_file = f"./data/coco/annotations/instances_train2017.json"
    out_path = f"{annotations_path}/memory/train2017_allClasses_length{memory_length}_semantic_v1.pkl"
    sample_semantic_ref_memory_dataset(all_refs_json_file, out_path, memory_length)
