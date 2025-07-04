import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import ParallelStrategy

from torch.optim.lr_scheduler import MultiStepLR

from dev_hongyi.models.SAM2Ref import SAM2Ref
from dev_hongyi.dataset.coco_ref_dataset import (
    COCORefTrainDataset,
    COCOMemoryFillDataset,
    COCORefTestDataset,
    COCORefOracleTestDataset
)
from dev_hongyi.utils import print_dict



class DummyDataset(Dataset):
    def __init__(self, length):
        super(DummyDataset, self).__init__()
        self.data = [1 for _ in range(length)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_dataset(dataset_cfg, stage):
    dataset_name = dataset_cfg.pop("name", None)
    assert dataset_name in ["coco"]
    assert stage in ["train", "fill_memory", "test"]

    if dataset_name == "coco":
        if stage == "train":
            return COCORefTrainDataset(**dataset_cfg)
        elif stage == "fill_memory":
            return COCOMemoryFillDataset(**dataset_cfg)
        elif stage == "test":
            return COCORefOracleTestDataset(**dataset_cfg)
        else:
            raise NotImplementedError("Unrecognized stage %s" % stage)
    else:
        raise NotImplementedError("Unrecognized dataset %s" % dataset_name)


class RefSam2LightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        train_cfg: dict,
        dataset_cfgs: dict,
        data_load_cfgs: dict,
        test_mode: str = "none"
    ):
        super().__init__()

        # training configurations
        self.weight_decay     = train_cfg.get("weight_decay")
        self.lr_decay_epochs  = train_cfg.get("lr_decay_epochs")
        self.warmup_iters     = train_cfg.get("warmup_iters")
        self.train_bs_per_gpu = train_cfg.get("train_bs_per_gpu")
        self.lr_cfg           = train_cfg.get("lr_cfg")

        # data configurations
        self.dataset_cfgs   = dataset_cfgs
        self.data_load_cfgs = data_load_cfgs
        self.workers        = data_load_cfgs.get("workers")

        self.test_mode = test_mode
        if test_mode != "none":
            model_cfg.update({"enable_memory_bank": True})
        self.model = SAM2Ref(**model_cfg)

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def load_state_dict(self, state_dict, strict=False, assign=False):
        super(RefSam2LightningModel, self).load_state_dict(state_dict, strict=False, assign=assign)

    def _output_inqueue(self, output_dict):
        results_per_img = dict(
            masks=output_dict["binaay_masks"].cpu().numpy(),
            boxes=output_dict["bboxes"].cpu().numpy(),
            scores=output_dict["scores"].cpu().numpy(),
            labels=output_dict["labels"].cpu().numpy(),
            img_id=int(output_dict["image_info"]["id"])
        )
        encoded_results_per_img = self.eval_dataset.encode_results([results_per_img])
        self.output_queue.append(encoded_results_per_img)

    def forward(self, x, return_iou_grid_scores=False):
        return self.model(x, return_iou_grid_scores)

    def training_step(self, batch, batch_idx):
        assert self.model.training
        loss_dict, metric_dict = self.model(batch)
        loss_total = sum(loss_dict.values())

        for k, v in metric_dict.items():
            self.log(k, v)

        for k, v in loss_dict.items():
            self.log(k, v)
        self.log("total_loss", loss_total)
        return loss_total

    def _inference_step(self, batch, batch_idx):
        assert not self.model.training

        if type(self.eval_dataset) is DummyDataset:
            return None

        if self.test_mode == "fill_memory":
            self.model(batch)
        elif self.test_mode == "test":
            output = self.model(batch)
            assert len(output) == len(batch)
            self._output_inqueue(output[0])
        else:
            raise NotImplementedError(
                "Test mode should be %s or %s, but got %s" % ("fill_memory", "test", self.test_mode)
            )
        return None

    def validation_step(self, batch, batch_idx):
        return self._inference_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            self._inference_step(batch, batch_idx)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure,):
        optimizer.step(closure=optimizer_closure)

        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.warmup_iters:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_iters))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        no_decay_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                    'norm' in name
                    or 'bn' in name
                    or 'ln' in name
                    or 'bias' in name
                    or 'pe' in name
                    or 'embed' in name
            ):
                no_decay_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.AdamW(
            [
                {'params': no_decay_params, 'weight_decay': 0.0},
                {'params': other_params, 'weight_decay': self.weight_decay}
            ],
            lr=self.lr
        )
        scheduler = MultiStepLR(optimizer, self.lr_decay_epochs, gamma=0.1)
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        # linearly scaling learning rate w.r.t. the total batch size
        num_processes = max(1, self.trainer.strategy.num_processes)
        total_bs = self.train_bs_per_gpu * num_processes
        self.lr = self.lr_cfg.get("base_lr") * total_bs / self.lr_cfg.get("base_bs")

        if stage == "fit":
            self.train_dataset = get_dataset(self.dataset_cfgs.get("train"), "train")
            self.eval_dataset = DummyDataset(length=16)
        elif stage == "validation":
            self.eval_dataset = DummyDataset(length=16)
        elif stage == "test" or stage == "predict":
            if self.test_mode not in ["fill_memory", "test"]:
                raise NotImplementedError(
                    "Test mode should be %s or %s, but got %s" % ("fill_memory", "test", self.test_mode)
                )
            if self.test_mode == "fill_memory":
                self.eval_dataset = get_dataset(self.dataset_cfgs.get("fill_memory"), "fill_memory")
            else:
                self.output_queue = []
                self.eval_dataset = get_dataset(self.dataset_cfgs.get("test"), "test")
        else:
            raise NotImplementedError("Unrecognized stage: %s" % stage)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_bs_per_gpu,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=lambda batch: batch
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=1,  # always use bs=1 for eval
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda batch: batch
        )

    def val_dataloader(self):
        return self.test_dataloader()