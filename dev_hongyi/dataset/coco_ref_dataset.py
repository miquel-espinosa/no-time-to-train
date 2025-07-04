import copy
import json
import pickle
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from sam2.utils.misc import _load_img_as_tensor

from dev_hongyi.dataset.metainfo import METAINFO

from tidecv import Data as TideData

def _load_image(
    img_path,
    image_size,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    normalize=True
):
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    image, height, width = _load_img_as_tensor(img_path, image_size)
    image = image.to(torch.float32)  # important to have consistent results downstream
    if normalize:
        image -= img_mean
        image /= img_std
    return image, height, width


def TideCOCOResult(results):
    """ Loads predictions from a COCO-style results file. """

    data = TideData('None')

    for res in results:
        image = res['image_id']
        _cls = res['category_id']
        score = res['score']
        box = res['bbox'] if 'bbox' in res else None
        mask = res['segmentation'] if 'segmentation' in res else None

        data.add_detection(image, _cls, score, box, mask)

    return data


class COCORefTrainDataset(data.Dataset):
    
    

    def __init__(
        self,
        root,
        json_file,
        image_size,
        remove_bad,
        max_cat_num,
        max_mem_length,
        n_pos_points,
        neg_ratio,
        semantic_ref=False,
        norm_img=True,
        class_split=None
    ):
        assert not (semantic_ref and remove_bad), "remove_bad should be disabled when using semantic_ref"
        self.root = root
        self.coco = COCO(json_file)

        self.image_size = image_size
        self.norm_img = norm_img
        self.n_pos_points = n_pos_points
        self.neg_ratio = neg_ratio
        self.max_cat_num = max_cat_num
        self.max_mem_length = max_mem_length
        self.semantic_ref = semantic_ref

        if class_split is None:
            self.cat_names = self.METAINFO['default_classes']
        else:
            self.cat_names = self.METAINFO[class_split]
        self.cat_ids = self.coco.getCatIds(catNms=self.cat_names)
        self.cat_ids_to_inds, self.cat_inds_to_ids = self._get_cat_inds(self.cat_ids)

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

    def _box_xyxy_to_xywh(self, boxes):
        if type(boxes) is torch.Tensor:
            new_boxes = torch.zeros_like(boxes)
        elif type(boxes) is np.ndarray:
            new_boxes = np.zeros_like(boxes)
        else:
            raise NotImplementedError("Unsupported boxes data type")

        new_boxes[..., 0] = boxes[..., 0]
        new_boxes[..., 1] = boxes[..., 1]
        new_boxes[..., 2] = boxes[..., 2] - boxes[..., 0]
        new_boxes[..., 3] = boxes[..., 3] - boxes[..., 1]
        return new_boxes

    def _get_cat_inds(self, cat_ids):
        cat_ids.sort()
        cat_ids_to_inds = OrderedDict()
        cat_inds_to_ids = OrderedDict()
        ind = 0
        for cid in cat_ids:
            cat_ids_to_inds[cid] = ind
            cat_inds_to_ids[ind] = cid
            ind += 1
        return cat_ids_to_inds, cat_inds_to_ids

    def _get_image_data(self, img_id, normalize, size=None):
        if size is None:
            image_size = self.image_size
        else:
            image_size = size
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image, _, _ = _load_image(img_path, image_size=image_size, normalize=normalize)
        return image, img_info

    def _sample_points(self, masks):
        mask_union = masks.max(dim=0)[0]
        pos_points = torch.argwhere(mask_union > 0)
        n_pos_total = len(pos_points)
        if n_pos_total == 0:
            raise ValueError("No positive points!")

        n_pos = min(n_pos_total, self.n_pos_points)
        sampled_pos = pos_points[torch.randperm(n_pos_total)[:n_pos]].flip(dims=(-1,))

        # pad negative points if positive points is not enough
        n_total_to_sample = int(self.n_pos_points * (self.neg_ratio + 1))
        n_neg = n_total_to_sample - n_pos
        neg_points = torch.argwhere(mask_union <= 0)
        n_neg_total = len(neg_points)
        sampled_neg = neg_points[torch.randperm(n_neg_total)[:n_neg]].flip(dims=(-1,))

        # maybe points are still not enough, pad them with randomly sampled points
        n_rest = n_total_to_sample - (sampled_neg.shape[0] + sampled_pos.shape[0])

        points_info = {}
        points_info['n_pos'] = n_pos
        points_info['n_neg'] = n_neg
        points_info['n_rest'] = n_rest

        if n_rest > 0:
            all_pos = torch.argwhere(mask_union > -999)
            sampled_rest = all_pos[torch.randperm(len(all_pos))[:n_rest]].flip(dims=(-1,))
            query_points = torch.cat((sampled_pos, sampled_neg, sampled_rest), dim=0)
        else:
            query_points = torch.cat((sampled_pos, sampled_neg), dim=0)
        return query_points, points_info

    def _load_resized_annotation(self, ann, width, height):
        mask = self.coco.annToMask(ann)
        mask = torch.from_numpy(mask).reshape(1, 1, height, width).to(torch.float)
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

    def _load_original_annotation(self, ann, width, height):
        mask = self.coco.annToMask(ann)
        mask = torch.from_numpy(mask).reshape(1, height, width).to(dtype=torch.float)

        bx1, by1, bw, bh = ann["bbox"]
        bx2 = bx1 + bw
        by2 = by1 + bh
        bbox = np.array([[bx1, by1, bx2, by2]])
        return mask, bbox

    def __getitem__(self, index):
        target_img_id = self.img_ids[index]

        tar_img, tar_img_info = self._get_image_data(target_img_id, normalize=self.norm_img)

        # To avoid OOM, only sample a subset of categories for training
        if len(self.img_to_cats[target_img_id]) < self.max_cat_num or self.max_cat_num < 0:
            tar_cats = self.img_to_cats[target_img_id]
        else:
            tar_cats_copy = copy.copy(self.img_to_cats[target_img_id])
            random.shuffle(tar_cats_copy)
            tar_cats = tar_cats_copy[:self.max_cat_num]

        # Load annotations
        tar_anns_by_cat = OrderedDict()
        for cat_id in tar_cats:
            cat_ind = self.cat_ids_to_inds[cat_id]
            tar_anns_by_cat[cat_ind] = {"masks_list": []}
        for ann in self.coco.loadAnns(self.img_to_anns[target_img_id]):
            if ann["category_id"] not in tar_cats:
                continue
            mask, _ = self._load_resized_annotation(ann, tar_img_info["width"], tar_img_info["height"])
            cat_ind = self.cat_ids_to_inds[ann["category_id"]]
            tar_anns_by_cat[cat_ind]["masks_list"].append(mask)

        # Stack annotations in the same category
        for cat_id in tar_cats:
            cat_ind = self.cat_ids_to_inds[cat_id]
            tar_anns_by_cat[cat_ind]["masks"] = torch.cat(tar_anns_by_cat[cat_ind]["masks_list"], dim=0)
            tar_anns_by_cat[cat_ind].pop("masks_list")
            query_points, points_info = self._sample_points(tar_anns_by_cat[cat_ind]["masks"])
            tar_anns_by_cat[cat_ind]["query_points"] = query_points
            tar_anns_by_cat[cat_ind]["points_info"] = points_info

        # Load references by category
        refs_by_cat = OrderedDict()
        for cat_id in tar_cats:
            # Randomly sample the number of reference images (to mimic testing)
            n_total_ref = random.choice(list(range(1, self.max_mem_length + 1)))
            cat_ind = self.cat_ids_to_inds[cat_id]

            refs_by_cat[cat_ind] = {"img_list": [], "mask_list": [], "img_info": []}

            # Load individual references
            n_ref = 0
            for i in np.random.permutation(len(self.cat_to_imgs_and_anns[cat_id])):
                ref_img_id, ref_ann_id = self.cat_to_imgs_and_anns[cat_id][i]
                if ref_img_id == target_img_id:
                    continue
                ref_img, ref_img_info = self._get_image_data(ref_img_id, normalize=self.norm_img)
                refs_by_cat[cat_ind]["img_list"].append(ref_img.unsqueeze(dim=0))

                if not self.semantic_ref:
                    ref_ann = self.coco.loadAnns([ref_ann_id])[0]
                    mask, _ = self._load_resized_annotation(ref_ann, ref_img_info["width"], ref_img_info["height"])
                    refs_by_cat[cat_ind]["mask_list"].append(mask)
                else:
                    ref_anns = self.coco.loadAnns(self.img_to_anns[ref_img_id])
                    masks_per_img = []
                    for ref_ann in ref_anns:
                        mask, _ = self._load_resized_annotation(ref_ann, ref_img_info["width"], ref_img_info["height"])
                        masks_per_img.append(mask.unsqueeze(dim=1))
                    refs_by_cat[cat_ind]["mask_list"].append(torch.cat(masks_per_img, dim=1).max(dim=1)[0])

                refs_by_cat[cat_ind]["img_info"].append(OrderedDict())
                refs_by_cat[cat_ind]["img_info"][-1]["ori_height"] = ref_img_info["height"]
                refs_by_cat[cat_ind]["img_info"][-1]["ori_width"] = ref_img_info["width"]
                refs_by_cat[cat_ind]["img_info"][-1]["file_name"] = ref_img_info["file_name"]
                refs_by_cat[cat_ind]["img_info"][-1]["id"] = ref_img_id

                n_ref += 1
                if n_ref >= n_total_ref:
                    break

            if n_ref == 0:
                raise ValueError("No reference!")

            # Stack references in the same category
            refs_by_cat[cat_ind]["imgs"] = torch.cat(refs_by_cat[cat_ind]["img_list"], dim=0)
            refs_by_cat[cat_ind]["masks"] = torch.cat(refs_by_cat[cat_ind]["mask_list"], dim=0)
            refs_by_cat[cat_ind].pop("img_list")
            refs_by_cat[cat_ind].pop("mask_list")

        # Gather data into a dict
        ret = OrderedDict()
        ret["data_mode"] = "train"
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



class COCOMemoryFillDataset(COCORefTrainDataset):
    def __init__(
        self,
        root,
        json_file,
        memory_pkl,
        image_size,
        memory_length,
        semantic_ref=False,
        norm_img=True,
        class_split=None,
        cat_names=[],
        custom_data_mode=None,
    ):
        super(COCORefTrainDataset, self).__init__()

        self.coco = COCO(json_file)

        with open(memory_pkl, 'rb') as f:
            self.sampled_memory_data = pickle.load(f)

        self.root = root
        self.image_size = image_size
        self.norm_img = norm_img
        self.memory_length = memory_length
        self.semantic_ref = semantic_ref

        if len(cat_names) > 0:
            self.cat_names = cat_names
        elif class_split is None:
            self.cat_names = self.METAINFO['default_classes']
        else:
            self.cat_names = self.METAINFO[class_split]
        self.cat_ids = self.coco.getCatIds(catNms=self.cat_names)
        self.cat_ids_to_inds, self.cat_inds_to_ids = self._get_cat_inds(self.cat_ids)

        for cat_id in self.sampled_memory_data.keys():
            n_ref = len(self.sampled_memory_data[cat_id])
            if n_ref != memory_length:
                raise ValueError(
                    "Category %d: Got %d references but the memory length is %d" % (cat_id, n_ref, memory_length)
                )

        self.all_data = []
        for cat_id in self.sampled_memory_data.keys():
            if cat_id not in self.cat_ids:
                continue
            for data_dict in self.sampled_memory_data[cat_id]:
                data_dict["category_id"] = cat_id
                self.all_data.append(data_dict)

        if custom_data_mode is None:
            self.data_mode = "fill_memory"
        else:
            self.data_mode = custom_data_mode


    def __getitem__(self, index):
        sampled_data = self.all_data[index]
        ref_img_id = sampled_data['img_id']
        ref_img, ref_img_info = self._get_image_data(ref_img_id, normalize=self.norm_img)
        cat_id = sampled_data['category_id']

        if not self.semantic_ref:
            ref_ann = self.coco.loadAnns(sampled_data['ann_ids'])[0]
            assert ref_ann['category_id'] == cat_id
            mask, _ = self._load_resized_annotation(ref_ann, ref_img_info["width"], ref_img_info["height"])
        else:
            masks_per_img = []
            for ref_ann in self.coco.loadAnns(sampled_data['ann_ids']):
                assert cat_id == ref_ann['category_id']
                mask, _ = self._load_resized_annotation(ref_ann, ref_img_info["width"], ref_img_info["height"])
                masks_per_img.append(mask.unsqueeze(dim=1))
            mask = torch.cat(masks_per_img, dim=1).max(dim=1)[0]

        cat_ind = self.cat_ids_to_inds[cat_id]
        refs_by_cat = OrderedDict()

        refs_by_cat[cat_ind] = {}
        refs_by_cat[cat_ind]["imgs"] = ref_img.unsqueeze(dim=0)
        refs_by_cat[cat_ind]["masks"] = mask
        refs_by_cat[cat_ind]["img_info"] = [OrderedDict()]
        refs_by_cat[cat_ind]["img_info"][-1]["ori_height"] = ref_img_info["height"]
        refs_by_cat[cat_ind]["img_info"][-1]["ori_width"] = ref_img_info["width"]
        refs_by_cat[cat_ind]["img_info"][-1]["file_name"] = ref_img_info["file_name"]
        refs_by_cat[cat_ind]["img_info"][-1]["id"] = ref_img_id

        ret = OrderedDict()
        ret["data_mode"] = self.data_mode
        ret["refs_by_cat"] = refs_by_cat
        return ret

    def __len__(self):
        return len(self.all_data)


class COCOMemoryFillCropDataset(COCOMemoryFillDataset):
    def __init__(
        self,
        context_ratio=0.1,
        custom_data_mode=None,
        *args,
        **kwargs
    ):
        super(COCOMemoryFillCropDataset, self).__init__(*args, **kwargs)
        self.context_ratio = context_ratio
        assert not self.semantic_ref
        if custom_data_mode is None:
            self.data_mode = "fill_memory"
        else:
            self.data_mode = custom_data_mode

    def __getitem__(self, index):
        assert not self.semantic_ref
        sampled_data = self.all_data[index]
        ref_img_id = sampled_data['img_id']
        ref_img_info = self.coco.loadImgs([ref_img_id])[0]
        img_h_ori, img_w_ori = ref_img_info["height"], ref_img_info["width"]

        # load image in original size
        ref_img, _ = self._get_image_data(ref_img_id, normalize=self.norm_img, size=(img_h_ori, img_w_ori))
        cat_id = sampled_data['category_id']

        ref_ann = self.coco.loadAnns(sampled_data['ann_ids'])[0]
        assert ref_ann['category_id'] == cat_id

        # annotation in original image size
        mask, bbox = self._load_original_annotation(ref_ann, img_w_ori, img_h_ori)
        assert len(mask) == 1 and len(mask) == len(bbox)

        x1, y1, x2, y2 = [int(x) for x in bbox[0].tolist()]
        box_w = x2 - x1
        box_h = y2 - y1

        # Method 1
        # crop_x1 = max(0, int(x1 - box_w * self.context_ratio * 0.5))
        # crop_y1 = max(0, int(y1 - box_h * self.context_ratio * 0.5))
        # crop_x2 = min(img_w_ori, int(x2 + box_w * self.context_ratio * 0.5))
        # crop_y2 = min(img_h_ori, int(y2 + box_h * self.context_ratio * 0.5))

        # Method 2: keep aspect ratio
        mid_x = (x1 + x2) * 0.5
        mid_y = (y1 + y2) * 0.5
        crop_size = max(box_w, box_h) * (1.0 + self.context_ratio)
        crop_x1 = max(0, int(mid_x - crop_size * 0.5))
        crop_y1 = max(0, int(mid_y - crop_size * 0.5))
        crop_x2 = min(img_w_ori, int(mid_x + crop_size * 0.5))
        crop_y2 = min(img_h_ori, int(mid_y + crop_size * 0.5))

        ref_img_crop = ref_img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        mask_crop = mask[:, crop_y1:crop_y2, crop_x1:crop_x2]

        ref_img_crop = F.interpolate(
            ref_img_crop.unsqueeze(dim=0),
            size=(self.image_size, self.image_size),
            mode='bicubic'
        )
        mask_crop = F.interpolate(
            mask_crop.unsqueeze(dim=0),
            size=(self.image_size, self.image_size),
            mode='bilinear'
        ).squeeze(dim=0)

        # ref_img_crop = ref_img_crop * mask_crop.unsqueeze(dim=0)

        cat_ind = self.cat_ids_to_inds[cat_id]
        refs_by_cat = OrderedDict()

        refs_by_cat[cat_ind] = {}
        refs_by_cat[cat_ind]["imgs"] = ref_img_crop
        refs_by_cat[cat_ind]["masks"] = mask_crop
        refs_by_cat[cat_ind]["img_info"] = [OrderedDict()]
        refs_by_cat[cat_ind]["img_info"][-1]["ori_height"] = ref_img_info["height"]
        refs_by_cat[cat_ind]["img_info"][-1]["ori_width"] = ref_img_info["width"]
        refs_by_cat[cat_ind]["img_info"][-1]["file_name"] = ref_img_info["file_name"]
        refs_by_cat[cat_ind]["img_info"][-1]["id"] = ref_img_id

        ret = OrderedDict()
        ret["data_mode"] = str(self.data_mode)
        ret["refs_by_cat"] = refs_by_cat
        return ret





class COCORefTestDataset(COCORefTrainDataset):
    def __init__(
        self,
        root,
        json_file,
        image_size,
        n_points_per_edge=16,
        norm_img=True,
        class_split=None,
        with_query_points=False,
        custom_data_mode=None,
        cat_names=[],
    ):
        super(COCORefTrainDataset, self).__init__()

        with open(json_file) as jf:
            coco_data_ori = json.load(jf)
            self.categories_ori = coco_data_ori['categories']
        self.ann_json_file = json_file

        if len(cat_names) > 0:
            self.cat_names = cat_names
        elif class_split is None:
            self.cat_names = self.METAINFO['default_classes']
        else:
            self.cat_names = self.METAINFO[class_split]

        self.class_split = class_split
        if self.class_split is None:
            self.class_split = "default_classes"

        if self.class_split != "default_classes":
            _coco = COCO(json_file)
            _cat_ids = _coco.getCatIds(catNms=self.cat_names)
            _ann_ids = _coco.getAnnIds(catIds=_cat_ids)
            _filtered_gt = _coco.loadAnns(_ann_ids)

            self.coco = COCO()
            self.coco.dataset = _coco.dataset.copy()
            self.coco.dataset['annotations'] = _filtered_gt
            self.coco.createIndex()
        else:
            self.coco = COCO(json_file)

        self.root = root
        self.img_ids = list(self.coco.imgs.keys())
        self.img_ids.sort()

        self.cat_ids = self.coco.getCatIds(catNms=self.cat_names)
        self.cat_ids_to_inds, self.cat_inds_to_ids = self._get_cat_inds(self.cat_ids)

        self.image_size = image_size
        self.norm_img = norm_img
        self.n_points_per_edge = n_points_per_edge
        self.with_query_points = with_query_points

        if custom_data_mode is None:
            self.data_mode = "test"
        else:
            self.data_mode = custom_data_mode

    def __getitem__(self, index):
        target_img_id = self.img_ids[index]

        tar_img, tar_img_info = self._get_image_data(target_img_id, normalize=self.norm_img)
        ret = OrderedDict()
        ret["data_mode"] = self.data_mode
        ret["target_img"] = tar_img
        ret["target_img_info"] = OrderedDict()
        ret["target_img_info"]["ori_height"] = tar_img_info["height"]
        ret["target_img_info"]["ori_width"] = tar_img_info["width"]
        ret["target_img_info"]["file_name"] = tar_img_info["file_name"]
        ret["target_img_info"]["id"] = target_img_id

        if self.with_query_points:
            x, y = np.meshgrid(
                np.linspace(0, self.image_size, self.n_points_per_edge),
                np.linspace(0, self.image_size, self.n_points_per_edge)
            )
            query_points = np.stack((x.reshape(-1), y.reshape(-1)), axis=-1)
            query_points += 0.5
            ret["query_points"] = torch.from_numpy(query_points)
        return ret

    def __len__(self):
        return len(self.img_ids)

    def encode_results(self, output_dicts):
        results = []
        for output_per_img in output_dicts:
            img_id = int(output_per_img["img_id"]) if str(output_per_img["img_id"]).isdigit() else output_per_img["img_id"]
            for i in range(len(output_per_img["scores"])):
                score = float(output_per_img["scores"][i])
                cat_id = int(self.cat_inds_to_ids[int(output_per_img["labels"][i])])

                box_xyxy = output_per_img["boxes"][i]
                box_xywh = self._box_xyxy_to_xywh(box_xyxy).tolist()

                mask_map = output_per_img["masks"][i]
                segmentation = mask_utils.encode(np.asfortranarray(mask_map.astype(np.uint8)))
                # Ensure segmentation is serializable
                segmentation['counts'] = segmentation['counts'].decode("utf-8")
                result = {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": box_xywh,
                    "score": score,
                    "segmentation": segmentation
                }
                results.append(result)
        return results


    def evaluate(self, results, output_result=False):
        coco_results = self.coco.loadRes(results)
        if output_result:
            # Save results into json file
            folder_name = 'results_jsons'
            file_name = 'coco_results.json'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(f"{folder_name}/{file_name}", "w") as f:
                json.dump(results, f)


        if self.class_split == "default_classes":
            from tidecv import TIDE
            import tidecv.datasets as tide_datasets

            tide_gt = tide_datasets.COCO(path=self.ann_json_file)
            tide_res = TideCOCOResult(results)
            tide = TIDE()
            tide.evaluate_range(tide_gt, tide_res, mode=TIDE.BOX)
            tide.summarize()
            tide.evaluate_range(tide_gt, tide_res, mode=TIDE.MASK)
            tide.summarize()


        
        cocoEval_bbox = COCOeval(self.coco, coco_results, "bbox")
        cocoEval_bbox.params.imgIds = self.img_ids
        cocoEval_bbox.evaluate()
        cocoEval_bbox.accumulate()
        cocoEval_bbox.summarize()

        cocoEval_segm = COCOeval(self.coco, coco_results, "segm")
        cocoEval_segm.params.imgIds = self.img_ids
        cocoEval_segm.evaluate()
        cocoEval_segm.accumulate()
        cocoEval_segm.summarize()


    def sample_negative(self, results, out_pkl, out_json, sample_num, score_thr=0.0):
        from dev_hongyi.dataset.data_utils import is_valid_annotation, get_false_positives

        coco_results = self.coco.loadRes(results)
        fp_results = {}
        for cat_id in self.cat_ids:
            fp_results[cat_id] = []

        res_anns = {}
        for ann_id, ann in coco_results.anns.items():
            if ann["category_id"] not in self.cat_ids:
                raise ValueError("Class id %d is not in dataset"%ann["category_id"])
            ann_img_id = ann['image_id']
            if ann_img_id not in res_anns.keys():
                res_anns[ann_img_id] = []
            res_anns[ann_img_id].append(ann)

        for img_id in res_anns.keys():
            results = res_anns[img_id]
            annotations = self.coco.loadAnns(self.img_to_anns[img_id])
            false_positives = get_false_positives(results, annotations, self.cat_ids, iou_thr=0.1, use_mask_iou=False)
            for cat_id in self.cat_ids:
                fp_results[cat_id].extend(false_positives[cat_id])

        for cat_id in self.cat_ids:
            if len(fp_results[cat_id]) < sample_num:
                raise RuntimeError("Category %d does not have enough false positives!"%cat_id)




        out_pkl_dict = {}
        out_json_dict = {}
        out_json_dict["images"] = []
        out_json_dict["categories"] = copy.deepcopy(self.categories_ori)
        out_json_dict["annotations"] = []
        image_id_in_store = []

        # Method 1: Random selection
        m = 0
        for cat_id in self.cat_ids:
            out_pkl_dict[cat_id] = []
            random.shuffle(fp_results[cat_id])
            n = 0

            sorted_inds = np.argsort(-1.0 * np.array([x["score"] for x in fp_results[cat_id]])).tolist()

            for j in sorted_inds:
                ann = fp_results[cat_id][j]
                img_info = coco_results.loadImgs([ann["image_id"]])[0]
                _ann = dict(
                    image_id=ann["image_id"],
                    category_id=cat_id,
                    segmentation=ann["segmentation"],
                    bbox=ann["bbox"],
                    id=int(m)
                )
                # if not is_valid_annotation(_ann, img_info):
                #     continue
                out_pkl_dict[cat_id].append({'img_id': ann["image_id"], 'ann_ids': [int(m)]})
                out_json_dict["annotations"].append(_ann)
                if ann["image_id"] not in image_id_in_store:
                    out_json_dict["images"].append(img_info)
                    image_id_in_store.append(ann["image_id"])
                n += 1
                m += 1
                if n >= sample_num:
                    break
            if n < sample_num:
                raise RuntimeError("Category %d does not have enough valid false positives!" % cat_id)






        with open(out_pkl, 'wb') as pw:
            pickle.dump(out_pkl_dict, pw)
        print("Sampled negative pickle file is output to: %s" % out_pkl)

        with open(out_json, 'w') as jw:
            json.dump(out_json_dict, jw)
        print("Sampled negative json file is output to: %s" % out_json)










class COCORefOracleTestDataset(COCORefTestDataset):
    def __init__(
        self,
        root,
        json_file,
        image_size,
        n_points_per_edge=16,
        norm_img=True,
        class_split=None,
        with_query_points=False,
        custom_data_mode=None,
        cat_names=[]
    ):
        super(COCORefOracleTestDataset, self).__init__(
            root, json_file, image_size, n_points_per_edge, norm_img, class_split, with_query_points, custom_data_mode, cat_names=cat_names
        )
        self.img_to_anns = {}

        for img_id in self.img_ids:
            self.img_to_anns[img_id] = []

        for ann_id, ann in self.coco.anns.items():
            if ann["category_id"] not in self.cat_ids:
                continue
            self.img_to_anns[ann['image_id']].append(ann_id)

    def __getitem__(self, index):
        output_dict = super(COCORefOracleTestDataset, self).__getitem__(index)
        target_img_id = self.img_ids[index]

        height = output_dict["target_img_info"]["ori_height"]
        width = output_dict["target_img_info"]["ori_width"]

        tar_anns_by_cat = OrderedDict()
        for ann in self.coco.loadAnns(self.img_to_anns[target_img_id]):
            mask, bboxes = self._load_resized_annotation(ann, width, height)
            cat_ind = self.cat_ids_to_inds[ann["category_id"]]
            if cat_ind not in tar_anns_by_cat.keys():
                tar_anns_by_cat[cat_ind] = {"masks_list": [], "bboxes_list": []}
            tar_anns_by_cat[cat_ind]["masks_list"].append(mask)
            tar_anns_by_cat[cat_ind]["bboxes_list"].append(bboxes)

        for cat_ind in tar_anns_by_cat.keys():
            tar_anns_by_cat[cat_ind]["masks"] = torch.cat(tar_anns_by_cat[cat_ind]["masks_list"], dim=0)
            tar_anns_by_cat[cat_ind]["bboxes"] = torch.cat(tar_anns_by_cat[cat_ind]["bboxes_list"], dim=0)
            tar_anns_by_cat[cat_ind].pop("masks_list")
            tar_anns_by_cat[cat_ind].pop("bboxes_list")

        output_dict["tar_anns_by_cat"] = tar_anns_by_cat
        return output_dict








