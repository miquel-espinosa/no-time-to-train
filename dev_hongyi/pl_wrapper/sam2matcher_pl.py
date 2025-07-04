import os
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule


from dev_hongyi.models.Sam2Matcher import Sam2Matcher
from dev_hongyi.models.Sam2MatchingBaseline import Sam2MatchingBaseline
from dev_hongyi.models.Sam2MatchingBaseline_noAMG import Sam2MatchingBaselineNoAMG

from dev_hongyi.dataset.metainfo import METAINFO

from dev_hongyi.dataset.coco_ref_dataset import (
    COCORefTrainDataset,
    COCOMemoryFillDataset,
    COCORefTestDataset,
    COCORefOracleTestDataset,
    COCOMemoryFillCropDataset
)
from dev_hongyi.utils import print_dict


class DummyDataset(Dataset):
    def __init__(self, length):
        super(DummyDataset, self).__init__()
        self.data = [0.0 for _ in range(length)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_dataset(dataset_cfg, stage):
    dataset_name = dataset_cfg.pop("name", None)
    assert dataset_name in ["coco"]
    assert stage in ["train", "fill_memory", "vis_memory", "test", "test_support", "fill_memory_neg"]

    if dataset_name == "coco":
        if stage == "train":
            raise NotImplementedError
        elif stage == "fill_memory":
            # return COCOMemoryFillDataset(**dataset_cfg)
            return COCOMemoryFillCropDataset(**dataset_cfg)
        elif stage == "vis_memory":
            dataset_cfg["custom_data_mode"] = "vis_memory"
            return COCOMemoryFillCropDataset(**dataset_cfg)
        elif stage == "fill_memory_neg":
            dataset_cfg["custom_data_mode"] = "fill_memory_neg"
            return COCOMemoryFillCropDataset(**dataset_cfg)
            # dataset_cfg.pop("context_ratio", None)
            # return COCOMemoryFillDataset(**dataset_cfg)
        elif stage == "test":
            return COCORefOracleTestDataset(**dataset_cfg)
        elif stage == "test_support":
            dataset_cfg["custom_data_mode"] = "test_support"
            return COCORefOracleTestDataset(**dataset_cfg)
        else:
            raise NotImplementedError("Unrecognized stage %s" % stage)
    else:
        raise NotImplementedError("Unrecognized dataset %s" % dataset_name)


class Sam2MatcherLightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        dataset_cfgs: dict,
        data_load_cfgs: dict,
        test_mode: str = "none"
    ):
        super().__init__()

        # data configurations
        self.dataset_cfgs   = dataset_cfgs
        self.data_load_cfgs = data_load_cfgs
        self.workers        = data_load_cfgs.get("workers")

        # HACK to allow CLI arguments to override model_cfg
        if "fill_memory.root" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["root"] = dataset_cfgs.pop("fill_memory.root")
        if "fill_memory.json_file" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["json_file"] = dataset_cfgs.pop("fill_memory.json_file")
        if "fill_memory.memory_pkl" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["memory_pkl"] = dataset_cfgs.pop("fill_memory.memory_pkl")
        if "fill_memory.memory_length" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["memory_length"] = int(dataset_cfgs.pop("fill_memory.memory_length"))
        if "fill_memory.cat_names" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["cat_names"] = dataset_cfgs.pop("fill_memory.cat_names").split(",")
        if "fill_memory.class_split" in dataset_cfgs:
            dataset_cfgs["fill_memory"]["class_split"] = dataset_cfgs.pop("fill_memory.class_split")
            dataset_cfgs["fill_memory"]["cat_names"] = METAINFO[dataset_cfgs["fill_memory"]["class_split"]]
        if "memory_bank_cfg.length" in model_cfg:
            model_cfg["memory_bank_cfg"]["length"] = int(model_cfg.pop("memory_bank_cfg.length"))
        if "memory_bank_cfg.category_num" in model_cfg:
            model_cfg["memory_bank_cfg"]["category_num"] = int(model_cfg.pop("memory_bank_cfg.category_num"))
        if "test.root" in dataset_cfgs:
            dataset_cfgs["test"]["root"] = dataset_cfgs.pop("test.root")
        if "test.json_file" in dataset_cfgs:
            dataset_cfgs["test"]["json_file"] = dataset_cfgs.pop("test.json_file")
        if "test.cat_names" in dataset_cfgs:
            dataset_cfgs["test"]["cat_names"] = dataset_cfgs.pop("test.cat_names").split(",")
        if "test.class_split" in dataset_cfgs:
            dataset_cfgs["test"]["class_split"] = dataset_cfgs.pop("test.class_split")
            dataset_cfgs["test"]["cat_names"] = METAINFO[dataset_cfgs["test"]["class_split"]]
        self.test_mode = test_mode
        self.model_cfg = copy.deepcopy(model_cfg)

        model_name = model_cfg.pop("name").lower()
        if model_name == "matcher":
            self.seg_model = Sam2Matcher(**model_cfg)
        elif model_name == "matching_baseline":
            self.seg_model = Sam2MatchingBaseline(**model_cfg)
        elif model_name == "matching_baseline_noamg":
            self.seg_model = Sam2MatchingBaselineNoAMG(**model_cfg)
        else:
            raise NotImplementedError(f"Unrecognized model name: {model_name}")

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def load_state_dict(self, state_dict, strict=False, assign=False):
        if state_dict is not None:
            super(Sam2MatcherLightningModel, self).load_state_dict(state_dict, strict=False, assign=assign)

    def _output_inqueue(self, output_dict):
        results_per_img = dict(
            masks=output_dict["binary_masks"].cpu().numpy(),
            boxes=output_dict["bboxes"].cpu().numpy(),
            scores=output_dict["scores"].cpu().numpy(),
            labels=output_dict["labels"].cpu().numpy(),
            img_id=int(output_dict["image_info"]["id"]) if str(output_dict["image_info"]["id"]).isdigit() else output_dict["image_info"]["id"]
            # img_id=int(output_dict["image_info"]["id"])
        )
        encoded_results_per_img = self.eval_dataset.encode_results([results_per_img])
        self.output_queue.append(encoded_results_per_img)

        score_to_analysis = output_dict.get("score_to_analysis", None)
        if score_to_analysis is not None:
            self.scalars_queue.append(score_to_analysis.cpu().numpy())

    def forward(self, x, return_iou_grid_scores=False):
        return self.seg_model(x, return_iou_grid_scores)

    def test_step(self, batch, batch_idx):
        assert not self.seg_model.training
        with torch.inference_mode():
            # For clarity, each mode is an independent case although some modes may share similar logistic
            if self.test_mode == "fill_memory":
                self.seg_model(batch)
            elif self.test_mode == "vis_memory":
                self.seg_model(batch)
            elif self.test_mode == "fill_memory_neg":
                self.seg_model(batch)
            elif self.test_mode == "test_support":
                output = self.seg_model(batch)
                assert len(output) == len(batch)
                self._output_inqueue(output[0])
            elif self.test_mode == "test":
                output = self.seg_model(batch)
                assert len(output) == len(batch)
                self._output_inqueue(output[0])
            elif self.test_mode == "postprocess_memory":
                self.seg_model.postprocess_memory()
            elif self.test_mode == "postprocess_memory_neg":
                self.seg_model.postprocess_memory_negative()
            else:
                raise NotImplementedError("Unrecognized test mode: %s" % self.test_mode)
            return None

    def setup(self, stage: str):
        if stage != "test" and stage != "predict":
            raise NotImplementedError

        # To store results
        self.output_queue = []
        self.scalars_queue = []

        # For clarity, each mode is an independent case although some modes may share similar logistic
        if self.test_mode == "fill_memory":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("fill_memory"), "fill_memory")
        elif self.test_mode == "vis_memory":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("fill_memory"), "vis_memory")
        elif self.test_mode == "test_support":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("support"), "test_support")
        elif self.test_mode == "fill_memory_neg":
            # NOTE: This fill_memory cfg is over-written in PL's before_test method
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("fill_memory"), "fill_memory_neg")
        elif self.test_mode == "test":
            self.eval_dataset = get_dataset(self.dataset_cfgs.get("test"), "test")
        elif self.test_mode == "postprocess_memory":
            self.eval_dataset = DummyDataset(1)
        elif self.test_mode == "postprocess_memory_neg":
            self.eval_dataset = DummyDataset(1)
        else:
            raise NotImplementedError("Unrecognized test mode: %s" % self.test_mode)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=1,  # Always use bs=1 for eval, TODO: remove this limitation
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda batch: batch
        )