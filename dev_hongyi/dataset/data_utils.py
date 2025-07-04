import pickle
import random
import json
from collections import OrderedDict

from pycocotools.coco import COCO

import numpy as np

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
        ),
    'few_shot_classes':
            (
                'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train',
                'bottle', 'chair', 'dining table', 'potted plant', 'bird', 'cat',
                'cow', 'dog', 'horse', 'sheep', 'person', 'couch',  'tv'
            )
}

def is_valid_annotation(ann, img_info):
    img_w, img_h = img_info["width"], img_info["height"]

    # ann = coco.loadAnns([ann_id])[0]
    if ann.get("iscrowd", 0) == 1:
        return False
    # If the bbox is too small, ignore it
    if (
            ann["bbox"][2] < 32
            or ann["bbox"][3] < 32
    ):
        return False
    # Ignore objects that are too far away from the image center
    if (
            ann["bbox"][0] < 10
            or ann["bbox"][1] < 10
            or img_w - (ann["bbox"][0] + ann["bbox"][2]) < 10
            or img_h - (ann["bbox"][1] + ann["bbox"][3]) < 10
    ):
        return False
    return True

def box_xywh_to_xyxy(bboxes):
    assert len(bboxes.shape) == 2 # [n, 4]
    new_bboxes = np.zeros_like(bboxes)
    new_bboxes[:, 0] += bboxes[:, 0]
    new_bboxes[:, 1] += bboxes[:, 1]
    new_bboxes[:, 2] += bboxes[:, 0] + bboxes[:, 2]
    new_bboxes[:, 3] += bboxes[:, 1] + bboxes[:, 3]
    return new_bboxes


def compute_box_iou_mat(bboxes1, bboxes2):
    n = bboxes1.shape[0]
    m = bboxes2.shape[0]

    _bboxes1 = np.tile(bboxes1[:, None, :], (1, m, 1))
    _bboxes2 = np.tile(bboxes2[None, :, :], (n, 1, 1))

    int_x1 = np.maximum(_bboxes1[:, :, 0], _bboxes2[:, :, 0])
    int_y1 = np.maximum(_bboxes1[:, :, 1], _bboxes2[:, :, 1])
    int_x2 = np.minimum(_bboxes1[:, :, 2], _bboxes2[:, :, 2])
    int_y2 = np.minimum(_bboxes1[:, :, 3], _bboxes2[:, :, 3])

    int_w = (int_x2 - int_x1).clip(min=0.0)
    int_h = (int_y2 - int_y1).clip(min=0.0)

    int_area = int_w * int_h
    area_1 = (_bboxes1[:, :, 2] - _bboxes1[:, :, 0]) * (_bboxes1[:, :, 3] - _bboxes1[:, :, 1])
    area_2 = (_bboxes2[:, :, 2] - _bboxes2[:, :, 0]) * (_bboxes2[:, :, 3] - _bboxes2[:, :, 1])

    iou = int_area / (area_1 + area_2 - int_area + 1e-10)
    return iou


def get_false_positives(results, annotations, cat_ids, iou_thr, use_mask_iou):
    false_positives = {}
    for cat_id in cat_ids:
        false_positives[cat_id] = []

    anns_by_cat = {}
    for ann in annotations:
        if ann['category_id'] not in cat_ids:
            continue
        if ann['category_id'] not in anns_by_cat.keys():
            anns_by_cat[ann['category_id']] = []
        anns_by_cat[ann['category_id']].append(ann)

    for res_ann in results:
        if res_ann['category_id'] not in cat_ids:
            raise RuntimeError("Unrecognized category id %d for the given cat_ids" % res_ann['category_id'])
        if res_ann['category_id'] not in anns_by_cat.keys():
            false_positives[res_ann['category_id']].append(res_ann)
            continue
        if not use_mask_iou:
            res_box = box_xywh_to_xyxy(np.array([res_ann["bbox"]]).reshape(1, 4))
            ann_bboxes = box_xywh_to_xyxy(np.array([x["bbox"] for x in anns_by_cat[res_ann['category_id']]]).reshape(-1, 4))
            iou_mat = compute_box_iou_mat(res_box, ann_bboxes)
            if iou_mat.max() < iou_thr:
                false_positives[res_ann['category_id']].append(res_ann)
        else:
            raise NotImplementedError

    return false_positives



