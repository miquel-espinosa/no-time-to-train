import copy
import random
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
import pickle
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from sam2.utils.misc import _load_img_as_tensor

def _load_image(img_path,
                image_size,
                img_mean=(0.485, 0.456, 0.406),
                img_std=(0.229, 0.224, 0.225)):
        
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        # image = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
        image, height, width = _load_img_as_tensor(img_path, image_size)
        image = image.to(torch.float32) # important to have consistent results downstream
        # normalize by mean and std
        image -= img_mean
        image /= img_std
        
        return image, height, width


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, image_size):
        self.root = root
        self.image_size = image_size
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
    
    def _get_data(self, ann_id):
        
        annotation = self.coco.anns[ann_id]
        img_id = annotation['image_id']
        
        img_info = self.coco.loadImgs(img_id)[0]
        image, height, width = _load_image(os.path.join(self.root, img_info['file_name']), image_size=self.image_size)
        
        return image, height, width, annotation, img_info
    
    def __getitem__(self, index):
        
        ann_id = self.ids[index]
        return self._get_data(ann_id) # image, annotation, img_info
    

    def __len__(self):
        return len(self.ids)
    
class CocoValidationDataset(data.Dataset):
    """COCO Validation Dataset compatible with previous forward pass code."""      
    def __init__(self, root, json, image_size, json_queries, root_queries):
        self.root = root
        self.root_queries = root_queries
        self.image_size = image_size
        self.coco_val = COCO(json)
        self.coco_queries = COCO(json_queries)
        self.val_ann_ids = list(self.coco_val.anns.keys())
        
        # Create a dictionary mapping categories to their largest non-impossible query annotation
        self.cat_to_query = {}
        for ann, img_info in zip(self.coco_queries.anns.values(), self.coco_queries.imgs.values()):
            cat_id = ann['category_id']
            area = ann['area']
            if cat_id not in self.cat_to_query:
                self.cat_to_query[cat_id] = (ann, img_info)
            elif area > self.cat_to_query[cat_id][0]['area']:
                self.cat_to_query[cat_id] = (ann, img_info)
        print("Dictionary cat_to_query created")
    
    def _get_image_data(self, img_id, root, filename):

        img_path = os.path.join(root, filename)
        image, _, _ = _load_image(img_path, image_size=self.image_size)
        
        return image
    
    def __getitem__(self, index):
        
        target_ann_id = self.val_ann_ids[index]
        target_ann = self.coco_val.loadAnns(target_ann_id)[0]
        target_cat_id = target_ann['category_id']
        target_img_info = self.coco_val.loadImgs([target_ann['image_id']])[0]
        target_img = self._get_image_data(target_ann['image_id'], self.root, target_img_info['file_name'])
        
        query_ann, query_img_info = self.cat_to_query[target_cat_id]
        query_img = self._get_image_data(query_ann['image_id'], self.root_queries, query_img_info['file_name'])
        
        target = (target_img, target_img_info, [target_ann], None)
        query = (query_img, query_img_info, [query_ann], None)
        
        return target, query
    

    def __len__(self):
        return len(self.val_ann_ids)    

class CocoQueryTargetDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, image_size):
        self.root = root
        self.image_size = image_size
        self.coco = COCO(json)
        
        # Dictionary to map image ids to their corresponding category ids
        # E.g. {img_id: [cat_id1, cat_id2, ...]}
        self.img_id_to_cat_ids = {img_id: list(set(ann['category_id'] for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)))) for img_id in self.coco.imgs.keys()}
        print("Dict img_id_to_cat_ids created")
        
        # Group annotation ids by image and category
        # Now, each item in the list is a tuple (img_id, cat_id, ann_ids),
        # where ann_ids is a list of annotation ids for the image and category
        self.ids = [(img_id, cat_id, self.coco.getAnnIds(imgIds=img_id, catIds=cat_id))
                    for img_id in self.coco.imgs.keys()
                    for cat_id in self.img_id_to_cat_ids[img_id]]
        print("Ids created")
        
        # Create a dictionary to map every image id to its bad annotations ids (i.e. all the annotations that have isimpossible=True)
        self.img_id_to_bad_ann_ids = {}
        for ann in self.coco.anns.values():
            if ann['image_id'] not in self.img_id_to_bad_ann_ids:
                self.img_id_to_bad_ann_ids[ann['image_id']] = []
            if ann['isimpossible'] == '1':
                self.img_id_to_bad_ann_ids[ann['image_id']].append(ann['id'])
        print("Dictionary for bad ann ids created")
        
    def _get_image_data(self, img_id):

        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image, _, _ = _load_image(img_path, image_size=self.image_size)
        
        return image, img_info
    
    def _get_annotations(self, cat_id, img_id):
        # Get all annotations with the category id and image id
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=cat_id))
        # Remove bad annotations
        annotations = [ann for ann in annotations if ann['id'] not in self.img_id_to_bad_ann_ids[img_id]]
        
        return annotations
    
    def __getitem__(self, index):
        
        tar_img_id, tar_cat_id, tar_ann_ids = self.ids[index]
        tar_img, tar_img_info = self._get_image_data(tar_img_id)
        tar_anns = self.coco.loadAnns(tar_ann_ids)
        target_bad_anns = self.img_id_to_bad_ann_ids[tar_img_id]
        # print("Loaded target image data")
        
        query_img_ids = self.coco.getImgIds(catIds=tar_cat_id) # Get all image ids that share the same target category
        np.random.shuffle(query_img_ids) # Shuffle the list
        
        query_anns = None
        query_img_id = None
        for qid in query_img_ids:
            if qid != tar_img_id:
                query_anns_aux = self._get_annotations(tar_cat_id, qid)
                if len(query_anns_aux) > 0:
                    query_img_id = qid
                    query_anns = query_anns_aux
                    break # We found a valid query image with valid annotations
        
        if query_anns is None:
            print(f"No query annotations found for target image {tar_img_id} with category {tar_cat_id}")
            return None
        
        query_img, query_img_info = self._get_image_data(query_img_id)
        query_bad_anns = self.img_id_to_bad_ann_ids[query_img_id]
        # print("Loaded query image data")
        
        target = (tar_img, tar_img_info, tar_anns, target_bad_anns)
        query = (query_img, query_img_info, query_anns, query_bad_anns)
        
        return (target, query)

    def __len__(self):
        return len(self.ids)


def convert_polygon_to_mask(polygon, width, height):
    rle = mask_utils.frPyObjects(polygon, height, width)
    binary_mask = mask_utils.decode(rle).squeeze()
    # If there are more than one channel, merge them into a single channel.
    # This means that if we have multiple polygons for the same annotation,
    # we merge them into a single mask.
    if binary_mask.ndim == 3:
        binary_mask = binary_mask.max(axis=2)
    return binary_mask

class COCORefTrainDataset(data.Dataset):

    # Reference: MMDetection
    METAINFO = {
        'default_classes':
            ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
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
             'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    }

    def __init__(self, root, json, image_size, remove_bad, max_cat_num, max_mem_size, n_pos_points, neg_ratio, cat_names=[]):
        self.root = root
        self.image_size = image_size
        self.coco = COCO(json)

        self.n_pos_points = n_pos_points
        self.neg_ratio = neg_ratio
        self.max_cat_num = max_cat_num
        self.max_mem_size = max_mem_size

        if len(cat_names) == 0:
            self.cat_ids = self.coco.getCatIds(catNms=self.METAINFO['default_classes'])
        else:
            self.cat_ids = self.coco.getCatIds(catNms=cat_names)

        self.img_ids = []
        self.img_to_anns = {}
        self.img_to_cats = {}
        self.cat_to_imgs_and_anns = {}
        for ann_id, ann in self.coco.anns.items():
            if ann["category_id"] not in self.cat_ids:
                continue
            if remove_bad and ann["isimpossible"] == 1:
                continue

            ann_img_id = ann['image_id']
            ann_cat_id = ann["category_id"]
            if ann_img_id not in self.img_to_anns:
                self.img_to_anns[ann_img_id] = []
                self.img_to_cats[ann_img_id] = []
                self.img_ids.append(ann_img_id)
            if ann_cat_id not in self.cat_to_imgs_and_anns:
                self.cat_to_imgs_and_anns[ann_cat_id] = []
            self.img_to_anns[ann_img_id].append(ann_id)
            if ann_cat_id not in self.img_to_cats[ann_img_id]:
                self.img_to_cats[ann_img_id].append(ann_cat_id)
            self.cat_to_imgs_and_anns[ann_cat_id].append((ann_img_id, ann_id))

    def _get_image_data(self, img_id):
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image, _, _ = _load_image(img_path, image_size=self.image_size)
        return image, img_info

    def _sample_points(self, masks):
        mask_union = masks.max(dim=0)[0]
        pos_points = torch.argwhere(mask_union > 0)
        n_pos_total = len(pos_points)
        if n_pos_total == 0:
            raise ValueError("No positive points!")

        n_pos = min(n_pos_total, self.n_pos_points)
        sampled_pos = pos_points[torch.randperm(n_pos_total)[:n_pos]]

        # pad negative points if positive points is not enough
        n_neg = int(self.n_pos_points * self.neg_ratio - n_pos)
        neg_points = torch.argwhere(mask_union <= 0)
        n_neg_total = len(neg_points)
        sampled_neg = neg_points[torch.randperm(n_neg_total)[:n_neg]]
        return sampled_pos, sampled_neg


    def _load_resized_annotation(self, ann, width, height):
        mask = convert_polygon_to_mask(ann["segmentation"], width, height)
        mask = torch.from_numpy(mask).reshape(1, 1, width, height)
        mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="nearest")
        mask = mask.squeeze(dim=0)

        bx1, by1, bw, bh = ann["bbox"]
        bx2 = bx1 + bw
        by2 = by1 + bh
        rbx1 = bx1 * self.image_size / width
        rbx2 = bx2 * self.image_size / width
        rby1 = by1 * self.image_size / height
        rby2 = by2 * self.image_size / height
        bbox = torch.tensor([rbx1, rby1, rbx2, rby2]).unsqueeze(dim=0)
        return mask, bbox


    def __getitem__(self, index):
        target_img_id = self.img_ids[index]

        tar_img, tar_img_info = self._get_image_data(target_img_id)

        if len(self.img_to_cats[target_img_id]) < self.max_cat_num or self.max_cat_num < 0:
            tar_cats = self.img_to_cats[target_img_id]
        else:
            tar_cats_copy = copy.copy(self.img_to_cats[target_img_id])
            random.shuffle(tar_cats_copy)
            tar_cats = tar_cats_copy[:self.max_cat_num]

        tar_anns_by_cat = OrderedDict()
        for cat_id in tar_cats:
            tar_anns_by_cat[cat_id] = {"masks_list": [], "bboxes_list": []}

        for ann in self.coco.loadAnns(self.img_to_anns[target_img_id]):
            if ann["category_id"] not in tar_cats:
                continue
            mask, bbox = self._load_resized_annotation(ann, tar_img_info["width"], tar_img_info["height"])
            tar_anns_by_cat[ann["category_id"]]["masks_list"].append(mask)
            tar_anns_by_cat[ann["category_id"]]["bboxes_list"].append(bbox)

        for cat_id in tar_cats:
            tar_anns_by_cat[cat_id]["masks"] = torch.cat(tar_anns_by_cat[cat_id]["masks_list"], dim=0)
            tar_anns_by_cat[cat_id]["bboxes"] = torch.cat(tar_anns_by_cat[cat_id]["bboxes_list"], dim=0)
            tar_anns_by_cat[cat_id].pop("masks_list")
            tar_anns_by_cat[cat_id].pop("bboxes_list")
            pos_points, neg_points = self._sample_points(tar_anns_by_cat[cat_id]["masks"])
            tar_anns_by_cat[cat_id]["pos_points"] = pos_points
            tar_anns_by_cat[cat_id]["neg_points"] = neg_points

        refs_by_cat = OrderedDict()
        for cat_id in tar_cats:
            n_total_ref = random.choice(list(range(1, self.max_mem_size+1)))
            refs_by_cat[cat_id] = {"img_list": [], "mask_list": [], "bbox_list": [], "img_info": []}

            n_ref = 0
            for i in np.random.permutation(len(self.cat_to_imgs_and_anns[cat_id])):
                ref_img_id, ref_ann_id = self.cat_to_imgs_and_anns[cat_id][i]
                if ref_img_id == target_img_id:
                    continue
                ref_img, ref_img_info = self._get_image_data(ref_img_id)
                ref_ann = self.coco.loadAnns([ref_ann_id])[0]
                mask, bbox = self._load_resized_annotation(ref_ann, ref_img_info["width"], ref_img_info["height"])
                refs_by_cat[cat_id]["img_list"].append(ref_img.unsqueeze(dim=0))
                refs_by_cat[cat_id]["mask_list"].append(mask)
                refs_by_cat[cat_id]["bbox_list"].append(bbox)
                refs_by_cat[cat_id]["img_info"].append(OrderedDict())
                refs_by_cat[cat_id]["img_info"][-1]["ori_height"] = ref_img_info["height"]
                refs_by_cat[cat_id]["img_info"][-1]["ori_width"] = ref_img_info["width"]
                refs_by_cat[cat_id]["img_info"][-1]["file_name"] = ref_img_info["file_name"]
                refs_by_cat[cat_id]["img_info"][-1]["id"] = ref_img_id

                n_ref += 1
                if n_ref >= n_total_ref:
                    break

            if n_ref == 0:
                raise ValueError("No reference!")

            refs_by_cat[cat_id]["imgs"] = torch.cat(refs_by_cat[cat_id]["img_list"], dim=0)
            refs_by_cat[cat_id]["masks"] = torch.cat(refs_by_cat[cat_id]["mask_list"], dim=0)
            refs_by_cat[cat_id]["bboxes"] = torch.cat(refs_by_cat[cat_id]["bbox_list"], dim=0)
            refs_by_cat[cat_id].pop("img_list")
            refs_by_cat[cat_id].pop("mask_list")
            refs_by_cat[cat_id].pop("bbox_list")

        ret = OrderedDict()
        ret["target_img"] = tar_img
        ret["target_img_info"] = OrderedDict()
        ret["target_img_info"]["ori_height"] = tar_img_info["height"]
        ret["target_img_info"]["ori_width"] = tar_img_info["width"]
        ret["target_img_info"]["file_name"] = tar_img_info["file_name"]
        ret["target_img_info"]["id"] = target_img_id

        ret["tar_anns_by_cat"] = tar_anns_by_cat
        ret["refs_by_cat"] = refs_by_cat
        return ret

    def __len__(self):
        return len(self.img_ids)



    
class CocoImageDataset(data.Dataset):
    """COCO Image Only Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, image_size):
        self.root = root
        self.image_size = image_size
        self.coco = COCO(json)
        self.image_ids = list(self.coco.imgs.keys())
    
    def __getitem__(self, index):
        
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        return _load_image(img_path, image_size=self.image_size)  # image, height, width
    

    def __len__(self):
        return len(self.image_ids)


if __name__ == "__main__":
    # coco = COCO("data/coco/annotations_refsam2/instances_train2017_tiny_filtered_by_0.6.json")
    # ann = coco.loadAnns([156])[0]
    # image = Image.open(os.path.join("data/coco/train2017/", coco.loadImgs([ann["image_id"]])[0]["file_name"]))
    # image = np.array(image.convert("RGB"))
    # h, w = image.shape[:2]
    #
    # import pycocotools.mask as mask_utils
    # def convert_polygon_to_mask(polygon, width, height):
    #     rle = mask_utils.frPyObjects(polygon, height, width)
    #     binary_mask = mask_utils.decode(rle).squeeze()
    #     # If there are more than one channel, merge them into a single channel.
    #     # This means that if we have multiple polygons for the same annotation,
    #     # we merge them into a single mask.
    #     if binary_mask.ndim == 3:
    #         binary_mask = binary_mask.max(axis=2)
    #     return binary_mask
    #
    # binary_mask = convert_polygon_to_mask(ann["segmentation"], w, h)
    # print(binary_mask.dtype)
    # print(binary_mask.shape)
    # print(image.shape)
    #
    #

    def print_dict(d, space=0):
        for k, v in d.items():
            if type(v) is int or type(v) is float or type(v) is str:
                print(" "*space, str(k) + ":", type(v), v)
            elif type(v) is np.ndarray or type(v) is torch.Tensor:
                print(" "*space, str(k) + ":", type(v), v.shape)
            elif type(v) is list or type(v) is tuple:
                print(" "*space, str(k) + ":", type(v), len(v))
            elif type(v) is dict or type(v) is OrderedDict:
                print(" "*space, str(k) + ":")
                print_dict(v, space+4)
            else:
                print(" "*space, type(v), "UNDEFINED FORMAT")


    dataset = COCORefTrainDataset(
        root="./data/coco/train2017",
        json="./data/coco/annotations_refsam2/instances_train2017_tiny_filtered_by_0.6.json",
        image_size=1024,
        remove_bad=True,
        max_cat_num=4,
        max_mem_size=6,
        n_pos_points=16,
        neg_ratio=3.0
    )

    # print_dict(dataset[0])

    def custom_collate_fn(batch):
        return batch
        # batch_dict = {}
        #
        # # Gather all keys
        # keys = batch[0].keys()
        #
        #
        # for key in keys:
        #     # Check if the item is something you don't want batched (like metadata or images)
        #     if key in ['non_batchable_item']:  # Replace with the actual keys
        #         batch_dict[key] = [d[key] for d in batch]
        #     else:
        #         batch_dict[key] = torch.stack([d[key] for d in batch])
        #
        # return batch_dict

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: batch)
    for batch in train_loader:
        print_dict(batch[2])
        break
