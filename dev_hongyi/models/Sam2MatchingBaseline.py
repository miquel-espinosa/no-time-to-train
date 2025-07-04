import copy
import random
from collections import OrderedDict

import ot
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import Normalize

from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import fill_holes_in_mask_scores, concat_points
from sam2.utils.amg import batched_mask_to_box

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


import dinov2.dinov2.models.vision_transformer as dinov2_vit
import dinov2.dinov2.utils.utils as dinov2_utils

from dev_hongyi.models.matching_baseline_utils import SAM2AutomaticMaskGenerator_MatchingBaseline
from dev_hongyi.models.model_utils import concat_all_gather
from dev_hongyi.utils import print_dict


encoder_predefined_cfgs = {
    "dinov2_large": dict(
        model_size="vit_large",
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1024
    )
}


class Sam2MatchingBaseline(nn.Module):
    def __init__(
        self,
        sam2_cfg_file,
        sam2_ckpt_path,
        sam2_amg_cfg,
        encoder_cfg,
        encoder_ckpt_path,
        memory_bank_cfg
    ):
        super(Sam2MatchingBaseline, self).__init__()

        # Models
        self.sam_model = build_sam2(sam2_cfg_file, sam2_ckpt_path)
        self.sam_amg = SAM2AutomaticMaskGenerator_MatchingBaseline(self.sam_model, **sam2_amg_cfg)

        encoder_name = encoder_cfg.pop("name")
        encoder_args = copy.deepcopy(encoder_predefined_cfgs.get(encoder_name))
        encoder_args.update(encoder_cfg)

        encoder_img_size = encoder_args.get("img_size")
        encoder_patch_size = encoder_args.get("patch_size")
        encoder_hw = encoder_img_size // encoder_patch_size
        self.encoder_h, self.encoder_w = encoder_hw, encoder_hw
        self.encoder_dim = encoder_args.pop("feat_dim")

        self.encoder_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.encoder = dinov2_vit.__dict__[encoder_args.pop("model_size")](**encoder_args)
        dinov2_utils.load_pretrained_weights(self.encoder, encoder_ckpt_path, "teacher")

        self.sam_amg.predictor.model.eval()
        self.encoder.eval()

        # Others
        memory_bank_cfg["feat_shape"] = (self.encoder_h * self.encoder_w, self.encoder_dim)
        self._init_memory_bank(memory_bank_cfg)

    def _compute_matched_iou_matrix(self, gt_masks, pred_masks):
        with torch.inference_mode():
            assert len(gt_masks) == pred_masks.shape[0]

            N = len(gt_masks)
            n_pix = gt_masks[0].shape[-2] * gt_masks[0].shape[-1]
            n_points, n_output = pred_masks.shape[1], pred_masks.shape[2]

            matched_ious = []
            for i in range(N):
                gt = gt_masks[i].reshape(1, -1, n_pix)  # [1, n_ins, n_pix]
                n_ins = gt.shape[0]
                pred = pred_masks[i].reshape(-1, 1, n_pix).expand(-1, n_ins, -1)  # [n_points * n_output, n_ins, n_pix]

                intersection = torch.logical_and(gt, pred).to(dtype=torch.float)
                union = torch.logical_or(gt, pred).to(dtype=torch.float)
                iou = intersection.sum(dim=-1) / union.sum(dim=-1)
                matched_iou = iou.max(dim=-1)[0]  # [n_points * n_output]
                matched_ious.append(matched_iou)
            return torch.cat(matched_ious, dim=0)

    def _init_memory_bank(self, memory_bank_cfg):
        if memory_bank_cfg.pop("enable"):
            self.mem_n_classes = memory_bank_cfg.get("category_num")
            self.mem_length = memory_bank_cfg.get("length")
            self.mem_feat_shape = memory_bank_cfg.get("feat_shape")

            assert len(self.mem_feat_shape) == 2
            _mem_n, _mem_c = self.mem_feat_shape

            self.register_buffer(
                "mem_fill_counts", torch.zeros((self.mem_n_classes,), dtype=torch.long)
            )
            self.register_buffer(
                "mem_feats", torch.zeros((self.mem_n_classes, self.mem_length, _mem_n, _mem_c))
            )
            self.register_buffer(
                "mem_masks", torch.zeros((self.mem_n_classes, self.mem_length, _mem_n))
            )
            self.register_buffer(
                "mem_feats_avg", torch.zeros((self.mem_n_classes, _mem_c))
            )
            self.avg_initialized = False
            self.has_memory_bank = True
        else:
            self.avg_initialized = False
            self.has_memory_bank = False

    def _get_mem_feats_avg(self):
        if not self.avg_initialized:
            mem_feats_avg = (
                    torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=(1, 2))
                    / self.mem_masks.sum(dim=(1, 2)).unsqueeze(dim=1)
            )
            self.mem_feats_avg += mem_feats_avg
            self.avg_initialized = False
        return self.mem_feats_avg

    def _forward_encoder(self, imgs, normalize=True):
        assert len(imgs.shape) == 4

        B = imgs.shape[0]
        imgs = self.encoder_transform(imgs)
        feats = self.encoder.forward_features(imgs)["x_prenorm"][:, 1:]
        if normalize:
            feats = F.normalize(feats, dim=-1, p=2)
        feats = feats.reshape(B, -1, self.encoder_dim)
        return feats

    def forward_fill_memory(self, input_dicts):
        with torch.inference_mode():
            assert len(input_dicts) == 1

            device = self.sam_model.device

            ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

            ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
            ref_feats = self._forward_encoder(ref_imgs, normalize=True).reshape(1, -1, self.encoder_dim)

            ref_masks = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_feats.dtype)
            ref_masks = F.interpolate(
                ref_masks.unsqueeze(dim=0),
                size=(self.encoder_h, self.encoder_w),
                mode="nearest"
            ).reshape(1, -1)

            cat_ind_tensor = torch.tensor([ref_cat_ind], dtype=torch.long, device=device).reshape(1, 1)
            cat_ind_all = concat_all_gather(cat_ind_tensor).reshape(-1).to(dtype=torch.long).detach()
            feats_all = concat_all_gather(ref_feats.contiguous())
            masks_all = concat_all_gather(ref_masks.contiguous())

            for i in range(cat_ind_all.shape[0]):
                fill_ind = self.mem_fill_counts[cat_ind_all[i]]
                self.mem_feats[cat_ind_all[i], fill_ind] += feats_all[i]
                self.mem_masks[cat_ind_all[i], fill_ind] += masks_all[i]
                self.mem_fill_counts[cat_ind_all[i]] += 1

            return {}

    def _get_oracle_iou(self, lr_masks_all, tar_anns_by_cat):
        n_masks = lr_masks_all.shape[0] // self.mem_n_classes

        lr_masks_all = lr_masks_all.reshape(self.mem_n_classes, n_masks, *lr_masks_all.shape[-2:])
        scores_oracle = torch.zeros((self.mem_n_classes, n_masks), device=lr_masks_all.device)

        for cat_ind in tar_anns_by_cat.keys():
            lr_masks_cat = lr_masks_all[cat_ind].reshape(1, n_masks, 1, *lr_masks_all.shape[-2:])
            gt_masks_cat = tar_anns_by_cat[cat_ind]["masks"].to(dtype=torch.float, device=lr_masks_all.device)
            gt_masks_cat = F.interpolate(
                gt_masks_cat.unsqueeze(dim=1),
                size=(lr_masks_cat.shape[-2], lr_masks_cat.shape[-1]),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).squeeze(dim=1).bool()
            matched_iou = self._compute_matched_iou_matrix([gt_masks_cat], lr_masks_cat > 0).reshape(n_masks)
            scores_oracle[cat_ind] += matched_iou
        scores_oracle = scores_oracle.reshape(self.mem_n_classes * n_masks)
        return scores_oracle

    def forward_test(self, input_dicts):
        assert self.has_memory_bank
        assert len(input_dicts) == 1

        device = self.sam_model.device

        tar_img = input_dicts[0]["target_img"]
        tar_img_np = tar_img.permute(1, 2, 0).cpu().numpy()
        tar_img = tar_img.unsqueeze(dim=0).to(device=device)

        tar_img = self.encoder_transform(tar_img)
        tar_feat = self._forward_encoder(tar_img, normalize=True).reshape(-1, self.encoder_dim)  # [N, C]

        self.sam_amg.predictor.set_image(tar_img_np)
        _, pred_ious, lr_masks = self.sam_amg.generate(tar_img_np)  # masks
        self.sam_amg.predictor.reset_predictor()

        n_masks = lr_masks.shape[0]
        masks_feat_size = F.interpolate(
            lr_masks.unsqueeze(dim=1),
            size=(self.encoder_h, self.encoder_w),
            mode="bilinear",
            align_corners=True,
            antialias=True
        ).reshape(n_masks, -1)

        masks_feat_size_bool = masks_feat_size > 0
        non_empty_inds = masks_feat_size_bool.sum(dim=1) > 0

        masks_feat_size_bool = masks_feat_size_bool[non_empty_inds]
        pred_ious = pred_ious[non_empty_inds]
        lr_masks = lr_masks[non_empty_inds]

        n_masks = masks_feat_size_bool.shape[0]

        tar_avg_feats_all = []
        for i in range(n_masks):
            tar_avg_feats_all.append(
                tar_feat[masks_feat_size_bool[i]].mean(dim=0, keepdim=True)
            )
        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)

        mem_avg_feats = self._get_mem_feats_avg()
        sim_mat = (mem_avg_feats @ tar_avg_feats.t() + 1.0) / 2.0  # [mem_n_categories, n_masks]

        # scores_all_class = sim_mat.flatten() * pred_ious.unsqueeze(dim=0).expand(self.mem_n_classes, -1).flatten()
        # scores_all_class = pred_ious.unsqueeze(dim=0).expand(self.mem_n_classes, -1).flatten()
        # scores_all_class = sim_mat.flatten()

        lr_masks_all_class = (
            lr_masks
            .unsqueeze(dim=0)
            .expand(self.mem_n_classes, -1, -1, -1)
            .reshape(self.mem_n_classes * n_masks, *lr_masks.shape[-2:])
        )
        # scores_all_class = self._get_oracle_iou(lr_masks_all_class, input_dicts[0]["tar_anns_by_cat"]).flatten()
        labels_all_class = (
            torch.arange(self.mem_n_classes, device=device)
            .unsqueeze(dim=1)
            .expand(-1, n_masks)
            .flatten()
        )

        keep_inds = torch.argsort(scores_all_class, descending=True)[:100]

        scores_out = scores_all_class[keep_inds]
        lr_masks_out = lr_masks_all_class[keep_inds]
        labels_out = labels_all_class[keep_inds]

        # resizing and converting to output format
        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]

        masks_out = F.interpolate(
            lr_masks_out.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=True,
            antialias=True
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_out)
        output_dict = dict(
            binary_masks=masks_out,
            bboxes=bboxes,
            scores=scores_out,
            labels=labels_out,
            image_info=input_dicts[0]["target_img_info"],
        )

        return [output_dict]


    def forward(self, input_dicts):
        data_mode = input_dicts[0].pop("data_mode", None)
        assert data_mode is not None
        if self.training:
            raise NotImplementedError
        else:
            if data_mode == "fill_memory":
                return self.forward_fill_memory(input_dicts)
            elif data_mode == "test":
                return self.forward_test(input_dicts)
            else:
                raise NotImplementedError(f"Unrecognized data mode during inference: {data_mode}")
