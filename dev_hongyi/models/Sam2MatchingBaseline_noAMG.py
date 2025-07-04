import copy
import random
from collections import OrderedDict

import ot
import cv2
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.distributed as dist
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

from dev_hongyi.models.matching_baseline_utils import kmeans, kmeans_decouple
from dev_hongyi.models.model_utils import concat_all_gather
from dev_hongyi.utils import print_dict
from dev_hongyi.models.matching_baseline_utils import vis_pca, vis_kmeans, fast_l2

encoder_predefined_cfgs = {
    "dinov2_large": dict(
        model_size="vit_large",
        img_size=518,
        patch_size=14,
        init_values=1e-5,
        ffn_layer='mlp',
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        feat_dim=1024
    )
}



class Sam2MatchingBaselineNoAMG(nn.Module):
    def __init__(
        self,
        sam2_cfg_file,
        sam2_ckpt_path,
        sam2_infer_cfgs,
        encoder_cfg,
        encoder_ckpt_path,
        memory_bank_cfg,
        dataset_name='coco'
    ):
        super(Sam2MatchingBaselineNoAMG, self).__init__()

        self.dataset_name = dataset_name
        self.points_per_side = sam2_infer_cfgs.get("points_per_side")
        self.testing_point_bs = sam2_infer_cfgs.get("testing_point_bs")
        self.iou_thr = sam2_infer_cfgs.get("iou_thr")
        self.num_out_instance = sam2_infer_cfgs.get("num_out_instance")
        self.nms_thr = sam2_infer_cfgs.get("nms_thr")
        self.kmeans_k = sam2_infer_cfgs.get("kmeans_k")
        self.n_pca_components = sam2_infer_cfgs.get("n_pca_components")
        self.cls_num_per_mask = sam2_infer_cfgs.get("cls_num_per_mask")

        self.with_negative_refs = sam2_infer_cfgs.get("with_negative_refs", False)

        # Models
        self.sam_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.predictor = build_sam2_video_predictor(sam2_cfg_file, sam2_ckpt_path)
        self.sam_img_size = 1024

        encoder_name = encoder_cfg.pop("name")
        encoder_args = copy.deepcopy(encoder_predefined_cfgs.get(encoder_name))
        # encoder_args.update(encoder_cfg)

        encoder_img_size = encoder_cfg.get("img_size")
        encoder_patch_size = encoder_cfg.get("patch_size")
        encoder_hw = encoder_img_size // encoder_patch_size

        self.encoder_h, self.encoder_w = encoder_hw, encoder_hw
        self.encoder_img_size = encoder_img_size
        self.encoder_patch_size = encoder_patch_size
        self.encoder_dim = encoder_args.pop("feat_dim")

        self.encoder_transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.encoder = dinov2_vit.__dict__[encoder_args.pop("model_size")](**encoder_args)
        dinov2_utils.load_pretrained_weights(self.encoder, encoder_ckpt_path, "teacher")

        self.predictor.eval()
        self.encoder.eval()

        # Others
        memory_bank_cfg["feat_shape"] = (self.encoder_h * self.encoder_w, self.encoder_dim)
        self._init_memory_bank(memory_bank_cfg)

        self._reset()

    def _init_memory_bank(self, memory_bank_cfg):
        assert memory_bank_cfg.pop("enable")

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
        self.register_buffer(
            "mem_feats_ins_avg", torch.zeros((self.mem_n_classes, self.mem_length, _mem_c))
        )
        self.register_buffer(
            "mem_feats_covariances", torch.zeros((self.mem_n_classes, _mem_c, _mem_c))
        )
        self.register_buffer(
            "mem_feats_centers", torch.zeros((self.mem_n_classes, self.kmeans_k, _mem_c))
        )
        self.register_buffer(
            "mem_ins_sim_avg", torch.zeros((self.mem_n_classes,))
        )
        self.register_buffer(
            "mem_pca_mean", torch.zeros((self.mem_n_classes, _mem_c))
        )
        self.register_buffer(
            "mem_pca_components", torch.zeros((self.mem_n_classes, self.n_pca_components, _mem_c))
        )
        self.register_buffer("mem_postprocessed", torch.zeros((1,), dtype=torch.bool))
        self.memory_ready = False

        if self.with_negative_refs:
            self.mem_length_negative = memory_bank_cfg.get("length_negative")
            self.register_buffer(
                "mem_fill_counts_neg", torch.zeros((self.mem_n_classes,), dtype=torch.long)
            )
            self.register_buffer(
                "mem_feats_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_n, _mem_c))
            )
            self.register_buffer(
                "mem_masks_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_n))
            )
            self.register_buffer(
                "mem_feats_avg_neg", torch.zeros((self.mem_n_classes, _mem_c))
            )
            self.register_buffer(
                "mem_feats_ins_avg_neg", torch.zeros((self.mem_n_classes, self.mem_length_negative, _mem_c))
            )
            self.register_buffer("mem_postprocessed_neg", torch.zeros((1,), dtype=torch.bool))
            self.memory_neg_ready = False

    def _reset(self):
        self.backbone_features = None
        self.backbone_hr_features = None

    def _compute_matched_iou_matrix(self, gt_masks, pred_masks):
        with torch.inference_mode():
            assert len(gt_masks) == pred_masks.shape[0]

            N = len(gt_masks)
            n_pix = gt_masks[0].shape[-2] * gt_masks[0].shape[-1]
            n_points, n_output = pred_masks.shape[1], pred_masks.shape[2]

            matched_ious = []
            machted_inds = []
            for i in range(N):
                gt = gt_masks[i].reshape(1, -1, n_pix)  # [1, n_ins, n_pix]
                n_ins = gt.shape[0]
                pred = pred_masks[i].reshape(-1, 1, n_pix)  # .expand(-1, n_ins, -1)  # [n_points * n_output, n_ins, n_pix]

                intersection = torch.logical_and(gt, pred).to(dtype=torch.float)
                union = torch.logical_or(gt, pred).to(dtype=torch.float)
                iou = intersection.sum(dim=-1) / union.sum(dim=-1)
                matched_iou, matched_ins_inds = iou.max(dim=-1) # [n_points * n_output], [n_points * n_output]
                matched_ious.append(matched_iou)
                machted_inds.append(matched_ins_inds)
        return torch.cat(matched_ious, dim=0), torch.cat(machted_inds, dim=0)

    def _compute_semantic_ios(self, masks_binary, labels, obj_sim, use_semantic=True, rank_score=True):
        n_masks = masks_binary.shape[0]
        masks = masks_binary.reshape(n_masks, -1).to(dtype=torch.float32)
        ios = torch.zeros((n_masks,), device=masks_binary.device, dtype=torch.float32)

        for cat_ind in range(self.mem_n_classes):
            select_idxs = (labels == cat_ind)
            _masks = masks[select_idxs]
            _obj_sim = obj_sim[select_idxs][:, select_idxs]
            n_cat = _masks.shape[0]
            if n_cat == 0:
                continue
            pos_num = _masks.sum(dim=-1).to(dtype=torch.float32)
            inter_num = _masks @ _masks.t()
            inter_num.fill_diagonal_(0.0)
            if rank_score:
                inter_num = torch.tril(inter_num, diagonal=0)
            _ios = (inter_num / pos_num[:, None]) # .max(dim=-1)[0]
            if use_semantic:
                _ios = _ios * _obj_sim
            _ios = _ios.max(dim=-1)[0]
            ios[select_idxs] += _ios
        return ios

    def _compute_ios_batched(self, masks_binary, labels, rank_score=True, batch_size=10):
        n_masks = masks_binary.shape[0]
        masks = masks_binary.reshape(n_masks, -1).to(dtype=torch.float32)
        ios = torch.zeros((n_masks,), device=masks_binary.device, dtype=torch.float32)
        for cat_ind in range(self.mem_n_classes):
            select_idxs = (labels == cat_ind)
            _masks = masks[select_idxs]
            n_cat = _masks.shape[0]
            if n_cat == 0:
                continue
            
            # Process in batches to avoid OOM
            for i in range(0, n_cat, batch_size):
                batch_end = min(i + batch_size, n_cat)
                batch_masks = _masks[i:batch_end]
                
                pos_num = batch_masks.sum(dim=-1).to(dtype=torch.float32)
                
                # Compute inter_num in sub-batches if needed
                inter_num = torch.zeros((batch_masks.shape[0], n_cat), device=masks.device)
                for j in range(0, n_cat, batch_size):
                    j_end = min(j + batch_size, n_cat)
                    inter_num[:, j:j_end] = batch_masks @ _masks[j:j_end].t()
                
                if rank_score:
                    inter_num = torch.tril(inter_num, diagonal=0)
                    
                _ios = inter_num.max(dim=-1)[0] / pos_num
                ios[select_idxs][i:batch_end] += _ios
        return ios

    def _compute_pca_scores(self, tar_feats, binary_masks):
        n_masks = binary_masks.shape[0]

        pca_mean = self.mem_pca_mean.unsqueeze(dim=0)
        pca_components = (
            F.normalize(self.mem_pca_components, p=2, dim=-1)
            .permute(0, 2, 1)
            .unsqueeze(dim=0)
        )   # [1, n_class, c, n_components]

        scores_all = []
        for i in range(n_masks):
            foreground_feats = (
                tar_feats[binary_masks[i]]
                .unsqueeze(dim=1)
                .expand(-1, self.mem_n_classes, -1)
            )   # [n_fore, n_class, c]
            centered_feats = F.normalize(foreground_feats - pca_mean, p=2, dim=-1).unsqueeze(dim=2)
            pca_scores = centered_feats @ pca_components
            pca_scores = (pca_scores.squeeze(dim=2) + 1.0) * 0.5
            pca_scores = pca_scores.max(dim=0, keepdim=True)[0].mean(dim=-1)
            scores_all.append(pca_scores)
        scores_all = torch.cat(scores_all, dim=0)
        return scores_all

    def _compute_query_points(self, tar_feats, matching_size, num_points):
        assert matching_size[0] * matching_size[1] >= num_points
        device = tar_feats.device

        matching_h, matching_w = matching_size
        c = self.encoder_dim

        x, y = torch.meshgrid(
            torch.linspace(0, matching_w - 1, matching_w),
            torch.linspace(0, matching_h - 1, matching_h)
        )
        x = x + 0.5
        y = y + 0.5
        all_points = torch.stack((x.reshape(-1) / matching_w, y.reshape(-1) / matching_h), dim=-1)
        all_points = all_points.to(device=device)

        _tar_feats = tar_feats.reshape(1, self.encoder_h, self.encoder_w, c).permute(0, 3, 1, 2)
        _tar_feats = F.interpolate(
            _tar_feats,
            size=(matching_h, matching_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=0).reshape(c, -1).t()
        _tar_feats = F.normalize(_tar_feats, p=2, dim=-1)  # [n, c]

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        sim = _tar_feats @ mem_feats_avg.t()
        _, top_inds = torch.topk(sim.max(dim=1)[0], k=num_points)
        query_points = all_points[top_inds].reshape(num_points, 2)
        return query_points

    def _compute_ambiguous_decay(self, sim_global, labels):
        assert self.cls_num_per_mask == 1
        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        mem_sim_mat = mem_feats_avg @ mem_feats_avg.t()
        mem_sim_mat = (mem_sim_mat + 1.0) * 0.5
        mem_sim_mat.fill_diagonal_(0.5)

        mem_sim_select = mem_sim_mat[labels]
        decay = -1.0 * (sim_global - mem_sim_select).clamp(min=0.0).pow(2.0)  # [n_mask, c]
        weights = (mem_sim_select - 0.5).clamp(min=0.0)
        decay = (decay * weights).sum(dim=-1) / (weights.sum(dim=-1) + 1e-10)

        decay = torch.exp(decay / 0.1**2)
        # weights = (mem_sim_select - 0.5).clamp(min=0.0)
        # decay = (decay * weights).sum(dim=-1) / (weights.sum(dim=-1) + 1e-10)
        return decay

    def _get_oracle_iou(self, lr_masks_all, tar_anns_by_cat, matching_size=None):
        n_masks = lr_masks_all.shape[0]
        lr_masks_all = lr_masks_all.reshape(1, n_masks, *lr_masks_all.shape[-2:])

        if matching_size is not None:
            assert lr_masks_all.shape[-2] == lr_masks_all.shape[-1]
            matching_h, matching_w = matching_size, matching_size
            lr_masks_all = F.interpolate(
                lr_masks_all,
                size=(matching_h, matching_w),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            ).squeeze(dim=1)
        else:
            matching_h, matching_w = lr_masks_all.shape[-2], lr_masks_all.shape[-1]
        lr_masks_all = lr_masks_all > 0

        scores_oracle = torch.zeros((self.mem_n_classes, n_masks), device=lr_masks_all.device)

        for cat_ind in tar_anns_by_cat.keys():
            gt_masks_cat = tar_anns_by_cat[cat_ind]["masks"].to(dtype=torch.float, device=lr_masks_all.device)
            gt_masks_cat = F.interpolate(
                gt_masks_cat.unsqueeze(dim=1),
                size=(matching_h, matching_w),
                mode="nearest"
            ).squeeze(dim=1).bool()
            matched_iou, _ = self._compute_matched_iou_matrix([gt_masks_cat], lr_masks_all)
            matched_iou = matched_iou.reshape(n_masks)
            scores_oracle[cat_ind] += matched_iou
        scores_oracle = scores_oracle.reshape(self.mem_n_classes, n_masks)
        return scores_oracle

    def _get_oracle_refine_prompts(
        self,
        lr_masks_all,
        pred_labels_all,
        tar_anns_by_cat,
        pool_size=7,
        mask_stride=4
    ):
        device = lr_masks_all.device
        n_masks = lr_masks_all.shape[0]

        matching_h, matching_w = lr_masks_all.shape[-2], lr_masks_all.shape[-1]
        lr_masks_all = lr_masks_all > 0

        start = 0.5 / pool_size
        end = 1.0 - start
        intervals = torch.linspace(start, end, pool_size).to(dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(intervals, intervals)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        refine_points = torch.zeros((n_masks, pool_size**2, 2), dtype=torch.float32, device=device)
        refine_labels = torch.zeros((n_masks, pool_size**2), dtype=torch.float32, device=device)
        do_refine = torch.zeros((n_masks,), dtype=torch.float32, device=device)

        for cat_ind in tar_anns_by_cat.keys():
            gt_masks_cat = tar_anns_by_cat[cat_ind]["masks"].to(dtype=torch.float, device=device)
            gt_masks_cat = F.interpolate(
                gt_masks_cat.unsqueeze(dim=1),
                size=(matching_h, matching_w),
                mode="nearest"
            ).squeeze(dim=1).bool()

            cat_matched_inds = pred_labels_all==cat_ind
            lr_masks_cat = lr_masks_all[cat_matched_inds]
            n_masks_cat = lr_masks_cat.shape[0]
            if n_masks_cat == 0:
                continue
            bboxes_cat = batched_mask_to_box(lr_masks_cat)  # [n_cat, 4]
            lr_masks_cat = lr_masks_cat.unsqueeze(dim=0)
            _, matched_ind = self._compute_matched_iou_matrix([gt_masks_cat], lr_masks_cat)
            matched_ind = matched_ind.reshape(n_masks_cat)

            matched_gt_masks = gt_masks_cat[matched_ind].reshape(n_masks_cat, -1)
            lr_masks_cat = lr_masks_cat.reshape(n_masks_cat, -1)
            is_correct = torch.logical_and(lr_masks_cat, matched_gt_masks).to(dtype=torch.float32)

            x1, y1, x2, y2 = bboxes_cat[:, 0:1], bboxes_cat[:, 1:2], bboxes_cat[:, 2:3], bboxes_cat[:, 3:4]
            xs = x1 + (x2 - x1) * grid_x  # [n_masks_cat, pool_size * pool_size]
            ys = y1 + (y2 - y1) * grid_y  # [n_masks_cat, pool_size * pool_size]

            xs = xs.clamp(min=0, max=matching_w-1)
            ys = ys.clamp(min=0, max=matching_h-1)
            sample_pos = ys.to(dtype=torch.long) * matching_h + xs.to(dtype=torch.long)  # [n_masks_cat, pool_size * pool_size]
            is_correct_sampled = torch.gather(is_correct, 1, sample_pos)   # [n_masks_cat, pool_size * pool_size]

            do_refine[cat_matched_inds] += 1
            refine_points[cat_matched_inds] += torch.stack((xs, ys), dim=-1) * mask_stride
            refine_labels[cat_matched_inds] += is_correct_sampled

        return refine_points, refine_labels, do_refine

    def _get_refine_prompts(
        self,
        tar_feats,
        lr_masks_all,
        pred_labels_all,
        pool_size=7,
        mask_stride=4,
        thr=0.6
    ):
        device = lr_masks_all.device
        n_masks = lr_masks_all.shape[0]
        mask_h, mask_w = lr_masks_all.shape[-2:]

        lr_masks_all = lr_masks_all > 0
        bboxes = batched_mask_to_box(lr_masks_all)

        start = 0.5 / pool_size
        end = 1.0 - start
        intervals = torch.linspace(start, end, pool_size).to(dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(intervals, intervals)
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)

        x1, y1, x2, y2 = bboxes[:, 0:1], bboxes[:, 1:2], bboxes[:, 2:3], bboxes[:, 3:4]
        xs = x1 + (x2 - x1) * grid_x  # [n_masks, pool_size * pool_size]
        ys = y1 + (y2 - y1) * grid_y  # [n_masks, pool_size * pool_size]
        sampled_points_lr = torch.stack((xs, ys), dim=-1)

        xs = xs.clamp(min=0, max=mask_w) / mask_w
        ys = ys.clamp(min=0, max=mask_h) / mask_h
        sampled_points_normed = torch.stack((xs, ys), dim=-1)

        grid = sampled_points_normed.reshape(1, 1, -1, 2)
        grid = (grid - 0.5) * 2.0  # normalise to [-1, 1]

        tar_feats = tar_feats.reshape(1, self.encoder_h, self.encoder_w, self.encoder_dim).permute(0, 3, 1, 2)
        sampled_feats = F.grid_sample(tar_feats, grid).reshape(n_masks, pool_size**2, -1)
        sampled_feats = F.normalize(sampled_feats, p=2, dim=-1)

        temp_feats = self.mem_feats_avg[pred_labels_all].unsqueeze(dim=-1)  # [n_masks, c]
        temp_feats = F.normalize(temp_feats, p=2, dim=1)

        sim = sampled_feats @ temp_feats
        sim = (sim.reshape(n_masks, pool_size**2) + 1) * 0.5  # [n_masks, pool_size**2]

        sampled_points = sampled_points_lr * mask_stride
        sampled_labels = sim > thr
        return sampled_points, sampled_labels

    def _forward_encoder(self, imgs):
        assert len(imgs.shape) == 4
        B = imgs.shape[0]

        x = self.encoder.prepare_tokens_with_masks(imgs)
        n_skip_tokens = 1 + self.encoder.num_register_tokens
        for i, blk in enumerate(self.encoder.blocks):
            if i < len(self.encoder.blocks) - 1:
                x = blk(x)
            else:
                x, attn = blk(x, ret_attn=True)
                attn = attn.mean(dim=1)[:, n_skip_tokens :, n_skip_tokens :]
        x = self.encoder.norm(x)
        last_attn = attn
        feats = x[:, n_skip_tokens :]
        feats = feats.reshape(B, -1, self.encoder_dim)
        return feats, last_attn

    def _forward_encoder_attn_roll(self, imgs):
        def _select_head(attn_ws, fancy=False):
            if not fancy:
                return attn_ws.mean(dim=1)
            else:
                importance = attn_ws.max(dim=-1)[0].mean(dim=-1)  # [B, H]
                importance = importance[..., None, None]  # [B, H, 1, 1]
                attn_ws = attn_ws * importance
                attn_ws = attn_ws.mean(dim=1)
                return attn_ws

        def _roll_attn(attn_ws, attn_ws_prev, fancy=False):
            if not fancy:
                return attn_ws
            else:
                if attn_ws_prev is None:
                    return attn_ws
                n = attn_ws.shape[1]

                eye = torch.eye(n, device=imgs.device)
                rolled_attn = (eye + attn_ws) @ (eye + attn_ws_prev)# .transpose(1, 2)
                # rolled_attn = attn_ws @ attn_ws_prev
                # print(rolled_attn.sum(dim=-1).mean())
                # print(rolled_attn.sum(dim=-1).mean())
                return rolled_attn

        assert len(imgs.shape) == 4
        B = imgs.shape[0]

        x = self.encoder.prepare_tokens_with_masks(imgs)
        n_skip_tokens = 1 + self.encoder.num_register_tokens
        attn = None
        for i, blk in enumerate(self.encoder.blocks):
            x, _attn = blk(x, ret_attn=True)
            _attn = _select_head(_attn, fancy=False)
            attn = _roll_attn(_attn, attn, fancy=True)
        x = self.encoder.norm(x)
        attn = attn[:, n_skip_tokens :, n_skip_tokens :]
        feats = x[:, n_skip_tokens:]
        feats = feats.reshape(B, -1, self.encoder_dim)
        return feats, attn

    def _forward_sam_decoder(
        self,
        backbone_features,
        sparse_embeddings,
        dense_embeddings,
        backbone_hr_features,
        multimask_output=True
    ):
        B = backbone_features.shape[0]
        device = backbone_features.device
        (
            low_res_multimasks,
            ious,
            _,
            _,
        ) = self.predictor.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=backbone_hr_features,
            return_iou_token_out=False,
            disable_custom_iou_embed=True,
            disable_mlp_obj_scores=True,
            output_all_masks=True,
        )

        n_pred = ious.shape[-1]
        assert n_pred == low_res_multimasks.shape[1]

        # We skip the SAM2's multimask_output but use the custom IoU to determine the output mask
        # TODO: add advanced mask postprocessing tricks in the sam2 auto mask generator
        if multimask_output:
            best_iou_inds = torch.argmax(ious[:, 1:], dim=-1) + 1
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds]
            scores = ious[batch_inds, best_iou_inds]
        else:
            low_res_masks = low_res_multimasks[:, 0]
            scores = ious[:, 0]
        return low_res_masks, scores

    def _compute_masks(
        self,
        backbone_features,
        backbone_hr_features,
        point_inputs
    ):
        '''
        Similar to SAM2Base._forward_sam_heads. Putting it here for easy customization
        '''
        B = backbone_features.size(0)
        device = self.predictor.device
        assert backbone_features.size(1) == self.predictor.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.predictor.sam_image_embedding_size
        assert backbone_features.size(3) == self.predictor.sam_image_embedding_size

        sam_point_coords = point_inputs["point_coords"]
        sam_point_labels = point_inputs["point_labels"]
        assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, scores = self._forward_sam_decoder(
            backbone_features,
            sparse_embeddings,
            dense_embeddings,
            backbone_hr_features,
            multimask_output=True
        )
        return low_res_masks, scores

    def _compute_masks_refine(
        self,
        point_inputs,
        boxes_inputs,
        mask_inputs
    ):
        assert self.backbone_features is not None
        assert self.backbone_hr_features is not None
        # assert mask_inputs is not None

        backbone_features = self.backbone_features  # [1, c, h, w]
        backbone_hr_features = self.backbone_hr_features  # each [1, c, h, w]

        B = point_inputs["point_coords"].size(0)
        device = self.predictor.device

        backbone_features = backbone_features.expand(B, -1, -1, -1)
        backbone_hr_features = [x.expand(B, -1, -1, -1) for x in backbone_hr_features]

        if point_inputs is not None:
            points = (point_inputs["point_coords"], point_inputs["point_labels"])
        else:
            points = None
        if mask_inputs is not None:
            mask_inputs = mask_inputs.reshape(B, 1, *mask_inputs.shape[-2:]).to(dtype=backbone_features.dtype)
        else:
            mask_inputs = None

        sparse_embeddings, dense_embeddings = self.predictor.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=mask_inputs,
        )

        low_res_masks, scores = self._forward_sam_decoder(
            backbone_features,
            sparse_embeddings,
            dense_embeddings,
            backbone_hr_features,
            multimask_output=True
        )
        return low_res_masks, scores

    def _forward_sam_multiscale(self, imgs, scales=(0.5, 0.75, 1.0)):
        assert len(imgs.shape) == 4
        assert imgs.shape[-2] == imgs.shape[-1]
        assert self.backbone_features is None
        assert self.backbone_hr_features is None

        device = imgs.device

        sam_input_size = imgs.shape[-2]
        points_per_side = self.points_per_side

        lr_masks_all, scores_all = [], []
        for scale in scales:
            self.backbone_features = None
            self.backbone_hr_features = None

            if scale == 1.0:
                hw = sam_input_size
                padded_imgs = imgs
            else:
                hw = int(scale * sam_input_size)
                resized_imgs = F.interpolate(imgs, size=(hw, hw), mode='bicubic')
                padded_imgs = torch.zeros_like(imgs)  # use SAM's original input size to avoid potential bugs
                padded_imgs[:, :, :hw, :hw] += resized_imgs

            x, y = torch.meshgrid(
                torch.linspace(0, hw - 1, points_per_side),
                torch.linspace(0, hw - 1, points_per_side)
            )
            query_points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
            query_points += 0.5
            query_points = query_points.to(device=device)

            lr_masks, scores, _ = self._forward_sam(self.sam_transform(padded_imgs), query_points, point_normed=False)
            mask_hw = lr_masks.shape[-2]
            valid_mask_hw = int(mask_hw * scale)
            lr_masks = lr_masks[:, :valid_mask_hw, :valid_mask_hw]
            if valid_mask_hw != mask_hw:
                lr_masks = F.interpolate(
                    lr_masks.unsqueeze(dim=1),
                    size=(mask_hw, mask_hw),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True
                ).squeeze(dim=1)
            lr_masks_all.append(lr_masks)
            scores_all.append(scores)

        return torch.cat(lr_masks_all, dim=0), torch.cat(scores_all, dim=0), None

    def _forward_sam(self, imgs, precomputed_points=None, point_normed=True):
        assert len(imgs.shape) == 4
        assert imgs.shape[-2] == imgs.shape[-1]
        assert self.backbone_features is None
        assert self.backbone_hr_features is None

        device = imgs.device

        sam_input_size = imgs.shape[-2]
        points_per_side = self.points_per_side
        testing_point_bs = self.testing_point_bs
        iou_thr = self.iou_thr

        # Prepare input
        if precomputed_points is None:
            x, y = torch.meshgrid(
                torch.linspace(0, sam_input_size-1, points_per_side),
                torch.linspace(0, sam_input_size-1, points_per_side)
            )
            query_points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
            query_points += 0.5
            query_points = query_points.to(device=device)
        else:
            if point_normed:
                query_points = precomputed_points * sam_input_size
            else:
                query_points = precomputed_points


        # forward model
        backbone_out = self.predictor.forward_image(imgs)
        _, img_vision_features, img_vision_pos_embeds, img_feat_sizes = (
            self.predictor._prepare_backbone_features(backbone_out)
        )

        img_feats = img_vision_features[-1].permute(1, 2, 0).reshape(1, -1, *img_feat_sizes[-1])
        self.backbone_features = img_feats
        img_feats = img_feats.expand(testing_point_bs, -1, -1, -1)

        hr_feats = [
            x.permute(1, 2, 0).reshape(1, -1, *s)
            for x, s in zip(img_vision_features[:-1], img_feat_sizes[:-1])
        ]
        self.backbone_hr_features = hr_feats
        hr_feats = [
            x.expand(testing_point_bs, -1, -1, -1) for x in hr_feats
        ]

        points = query_points.reshape(-1, 2)
        point_labels = torch.ones_like(points[:, 0:1]).to(dtype=torch.int32)
        n_points = points.shape[0]

        mask_scores = []
        lr_masks = []
        for i in range(0, n_points // testing_point_bs):
            i_start = i * testing_point_bs
            i_end = i_start + testing_point_bs
            points_i = points[i_start:i_end, :]
            p_labels_i = point_labels[i_start:i_end, :]
            point_inputs_i = dict(
                point_coords=points_i.reshape(testing_point_bs, 1, 2),
                point_labels=p_labels_i.reshape(testing_point_bs, 1)
            )
            lr_masks_i, scores_i = self._compute_masks(
                img_feats, hr_feats, point_inputs_i
            )
            mask_scores.append(scores_i.reshape(-1))
            lr_masks.append(lr_masks_i.reshape(-1, *lr_masks_i.shape[-2:]))
        scores_all = torch.cat(mask_scores, dim=0).reshape(-1)
        lr_masks_all = torch.cat(lr_masks, dim=0)
        lr_masks_all = lr_masks_all.reshape(-1, *lr_masks_all.shape[-2:])

        inds = scores_all > iou_thr
        points_all = points[inds]
        lr_masks_all = lr_masks_all[inds]
        scores_all = scores_all[inds]

        return lr_masks_all, scores_all, points_all

    def forward_fill_memory(self, input_dicts, is_positive):
        with torch.inference_mode():
            assert len(input_dicts) == 1

            device = self.predictor.device

            ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

            ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
            ref_masks = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_imgs.dtype)

            ref_imgs = F.interpolate(
                ref_imgs,
                size=(self.encoder_img_size, self.encoder_img_size),
                mode="bicubic"
            )
            ref_imgs = self.encoder_transform(ref_imgs)
            ref_feats, _ = self._forward_encoder(ref_imgs)
            ref_feats = ref_feats.reshape(1, -1, self.encoder_dim)

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
                if is_positive:
                    if dist.is_initialized():
                        assert (self.mem_n_classes * self.mem_length) % dist.get_world_size() == 0
                    fill_ind = self.mem_fill_counts[cat_ind_all[i]]
                    self.mem_feats[cat_ind_all[i], fill_ind] += feats_all[i]
                    self.mem_masks[cat_ind_all[i], fill_ind] += masks_all[i]
                    self.mem_fill_counts[cat_ind_all[i]] += 1
                else:
                    if dist.is_initialized():
                        assert (self.mem_n_classes * self.mem_length_negative) % dist.get_world_size() == 0
                    fill_ind = self.mem_fill_counts_neg[cat_ind_all[i]]
                    self.mem_feats_neg[cat_ind_all[i], fill_ind] += feats_all[i]
                    self.mem_masks_neg[cat_ind_all[i], fill_ind] += masks_all[i]
                    self.mem_fill_counts_neg[cat_ind_all[i]] += 1

            return {}

    def forward_vis_memory(self, input_dicts):
        assert len(input_dicts) == 1
        assert self.mem_fill_counts[0].item() > 0
        assert self.n_pca_components == 3  # RGB

        device = self.predictor.device
        output_dir = "./results_analysis/memory_vis"

        ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

        ref_imgs = input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=device)
        ref_imgs = F.interpolate(
            ref_imgs,
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        ref_imgs_normed = self.encoder_transform(ref_imgs)
        ref_feats, _ = self._forward_encoder(ref_imgs_normed)
        ref_feats = ref_feats.reshape(-1, self.encoder_dim)

        ref_masks_ori = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_feats.dtype, device=device)
        ref_masks = F.interpolate(
            ref_masks_ori.unsqueeze(dim=0),
            size=(self.encoder_h, self.encoder_w),
            mode="nearest"
        ).reshape(-1)

        encoder_shape_info = dict(
            height=self.encoder_h,
            width=self.encoder_w,
            patch_size=self.encoder_patch_size
        )

        pca_vis_result = vis_pca(
            ref_imgs,
            ref_masks_ori,
            ref_cat_ind,
            ref_feats,
            ref_masks,
            self.mem_pca_mean,
            self.mem_pca_components,
            encoder_shape_info,
            device,
            transparency=1.0
        )
        kmeans_vis_result = vis_kmeans(
            ref_imgs,
            ref_masks_ori,
            ref_cat_ind,
            ref_feats,
            ref_masks,
            self.mem_feats_centers,
            encoder_shape_info,
            device,
            transparency=1.0
        )
        ori_img = ref_imgs[0].permute(1, 2, 0) * 255.0
        margin = torch.zeros((ori_img.shape[0], 5, 3), dtype=ori_img.dtype, device=device) + 255
        output_final = torch.cat((
            ori_img, margin, kmeans_vis_result, margin, pca_vis_result
        ), dim=1)

        import os
        from PIL import Image

        out_vis_img = Image.fromarray(output_final.cpu().numpy().astype(np.uint8))
        img_id = int(input_dicts[0]["refs_by_cat"][ref_cat_ind]["img_info"][0]['id'])
        out_vis_img.save(os.path.join(output_dir, "%d_%d.png" % (ref_cat_ind, img_id)))
        return {}

    def _compute_sim_attn_guided_global_avg(self, tar_feat, attn_weights, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        # bboxes = batched_mask_to_box(masks_feat_size_bool.reshape(-1, self.encoder_h, self.encoder_h))
        # box_size = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # box_scale = box_size / (self.encoder_h * self.encoder_w)

        attn_weights = attn_weights.reshape(tar_feat.shape[0], tar_feat.shape[0])
        tar_avg_feats_all = []
        tar_sizes = []
        hit_ratios = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            fore_attn = attn_weights[masks_feat_size_bool[i]]

            hit_attn = fore_attn[:, masks_feat_size_bool[i]]
            hit_ratios.append(hit_attn.sum() / fore_attn.sum())

            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))
            tar_sizes.append(torch.ones_like(feats_i[:, 0]).sum())
            # avg_feat = (feats_i * hit_ratio.unsqueeze(dim=1)).sum(dim=0) / hit_ratio.sum()
            # tar_avg_feats_all.append(avg_feat.unsqueeze(dim=0))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        tar_sizes = torch.stack(tar_sizes).reshape(-1, 1)
        tar_scale = tar_sizes / (self.encoder_w * self.encoder_h)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        sim_avg = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_avg = sim_avg.clamp(min=0.0)

        hit_ratios = torch.stack(hit_ratios).reshape(n_masks, 1)
        sigma = (1 - tar_scale) * 3.0
        # sigma = 2.0
        sim_avg = sim_avg * torch.exp(-(1 - hit_ratios)**2 / sigma**2)
        return sim_avg

    def _compute_sim_attn_guided_global_weighted(self, tar_feat, attn_weights, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        attn_weights = attn_weights.reshape(tar_feat.shape[0], tar_feat.shape[0])
        tar_avg_feats_all = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            fore_attn = attn_weights[masks_feat_size_bool[i]][:, masks_feat_size_bool[i]]

            importance = fore_attn.mean(dim=1).unsqueeze(dim=1)

            feats_i_avg = (feats_i * importance).sum(dim=0) / importance.sum()
            tar_avg_feats_all.append(feats_i_avg.unsqueeze(dim=0))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        sim_avg = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_avg = sim_avg.clamp(min=0.0)
        return sim_avg

    def _compute_sim_gaussian(self, tar_feat, masks_feat_size_bool):
        n_classes = self.mem_n_classes
        mem_feats = self.mem_feats_ins_avg  # [n_class, n_ins, c]

        mu = mem_feats.mean(dim=1)  # [n_class, c]

        feats_centered = mem_feats - mu.unsqueeze(dim=1)  # [n_class, n_ins, c]
        sigma = feats_centered.transpose(-1, -2) @ feats_centered / float(self.mem_length)  # [n_class, c, c]


        inv_sigma = torch.linalg.inv(sigma)  # [n_class, c, c]
        #
        # inverse_det_sigma = 1. / torch.sqrt(torch.det(sigma))  # [n_class]
        # print(torch.det(sigma))
        # exit()
        # inverse_det_sigma = inverse_det_sigma.unsqueeze(dim=0)

        n_masks = masks_feat_size_bool.shape[0]
        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]

        scores_all = []
        for i in range(n_classes):
            x_centered = x - mu[i:i+1, :]  # [n_masks, c]
            x_centered = x_centered.unsqueeze(dim=1)  # [n_masks, 1, c]
            score = torch.exp(-0.5 * (x_centered @ inv_sigma[i:i+1] @ x_centered.transpose(-1, -2)))
            scores_all.append(score.reshape(n_masks, 1))
        scores_all = torch.cat(scores_all, dim=1) # * inverse_det_sigma

        return scores_all.reshape(n_masks, n_classes)

    def _compute_sim_global_avg(self, tar_feat, masks_feat_size_bool, softmax=False, temp=1.0, ret_feats=False):
        n_masks = masks_feat_size_bool.shape[0]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        # mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        # mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        #
        mem_feats_avg = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, n_ins, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        sim_avg = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        if softmax:
            sim_avg = torch.softmax(sim_avg / temp, dim=-1)
        else:
            sim_avg = sim_avg.clamp(min=0.0)
        if not ret_feats:
            return sim_avg
        else:
            return sim_avg, tar_avg_feats

    def _compute_sim_global_l2(self, tar_feat, masks_feat_size_bool, sigma=1.0):
        n_masks = masks_feat_size_bool.shape[0]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)

        mem_feats_ins = self.mem_feats_ins_avg  # [n_class, n_ins, c]
        ins_dist = fast_l2(mem_feats_ins, mem_feats_ins, sqrt=True)
        ins_variance = ins_dist.reshape(self.mem_n_classes, -1).mean(dim=1)

        mem_feats_avg = self.mem_feats_avg   # [n_class, c]

        dist = fast_l2(tar_avg_feats, mem_feats_avg, sqrt=True)
        # norm = torch.pow(ins_variance, 0.15).unsqueeze(dim=0)
        # dist = dist / norm
        scores = torch.exp(-dist / sigma)
        return scores

    def _compute_completeness_decay(self, tar_feat, masks_feat_size_bool, global_sim, labels, decay=0.6):
        n_masks = masks_feat_size_bool.shape[0]

        tar_feat_normed = F.normalize(tar_feat, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        completenesses = []
        for i in range(n_masks):
            feats_i = tar_feat_normed[masks_feat_size_bool[i]]
            template_i = mem_feats_avg[labels[i]].unsqueeze(dim=1)

            sim_i = feats_i @ template_i
            sim_i = sim_i.clamp(min=0.0).flatten()

            thr_i = sim_i.max() * decay
            completeness = (sim_i > thr_i).sum() / torch.ones_like(sim_i).sum()
            completenesses.append(completeness)

        completenesses = torch.stack(completenesses).flatten()
        return completenesses

    def _compute_unification_decay(self, tar_feat, masks_feat_size_bool):
        device = tar_feat.device
        n_masks = masks_feat_size_bool.shape[0]

        # tar_feat_normed = F.normalize(tar_feat, p=2, dim=-1)

        unificationesses = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            if feats_i.shape[0] <= 1:
                unificationesses.append(torch.ones((1,), device=device))
                continue

            centers_i = kmeans(feats_i, k=2, n_iter=50)
            centers_i_normed = F.normalize(centers_i, p=2, dim=-1)
            center_sim = centers_i_normed @ centers_i_normed.t()
            unificationess = center_sim[0, 1]
            unificationesses.append(unificationess.reshape(1))

        unificationesses = torch.stack(unificationesses).flatten()
        return unificationesses

    def _compute_negative_decay(self, tar_feat, masks_feat_size_bool, sim_pos, labels):
        n_masks = masks_feat_size_bool.shape[0]
        c = tar_feat.shape[-1]

        tar_avg_feats_all = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_ins_avg_neg = self.mem_feats_ins_avg_neg[labels]  # [n_masks, n_ins, c]
        mem_feats_ins_avg_neg = F.normalize(mem_feats_ins_avg_neg, p=2, dim=-1)

        sim_neg = tar_avg_feats.unsqueeze(dim=1) @ mem_feats_ins_avg_neg.transpose(-1, -2)  # [n_masks, n_ins]
        sim_neg = sim_neg.clamp(min=0.0).squeeze(dim=1).max(dim=-1)[0]
        return sim_neg

    def _compute_kmeans_decay(self, tar_feat, masks_feat_size_bool):
        pass

    def _compute_sim_global_avg_with_neg(self, tar_feat, masks_feat_size_bool, sigma=1.0):
        n_masks = masks_feat_size_bool.shape[0]
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        tar_avg_feats = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        # mem_feats_avg = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, n_ins, c]
        # mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)

        mem_feats_ins_avg_neg = self.mem_feats_ins_avg_neg  # [n_class, n_ins, c]
        mem_feats_ins_avg_neg = F.normalize(mem_feats_ins_avg_neg, p=2, dim=-1).reshape(-1, c)
        
        pos_neg_sim = mem_feats_avg.unsqueeze(dim=1) @ mem_feats_ins_avg_neg.reshape(self.mem_n_classes, -1, c).transpose(-1,- 2)
        pos_neg_sim = pos_neg_sim.reshape(self.mem_n_classes, self.mem_length_negative)  # [n_class, n_ins]

        sim_pos = tar_avg_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_pos = sim_pos.clamp(min=0.0)

        sim_neg = tar_avg_feats @ mem_feats_ins_avg_neg.t()
        sim_neg = sim_neg.clamp(min=0.0)
        sim_neg = sim_neg.reshape(n_masks, self.mem_n_classes, -1)
        sim_neg, max_inds = sim_neg.max(dim=-1)

        # pos_neg_sim_selected = []
        # _arr_inds = torch.arange(self.mem_n_classes, device=tar_feat.device)
        # for i in range(n_masks):
        #     pos_neg_sim_selected.append(
        #         pos_neg_sim[_arr_inds, max_inds[i]].unsqueeze(dim=0)
        #     )  # [n_class]
        # pos_neg_sim_selected = torch.cat(pos_neg_sim_selected, dim=0)  # [n_masks, n_class]

        # sim_final = sim_pos * torch.exp(-1.0 * (sim_neg / (sim_pos+1e-10)).clamp(min=0.0) / sigma)
        sim_final = sim_pos * torch.exp(-1.0 * (sim_neg - sim_pos).clamp(min=0.0) / sigma)
        # sim_final = sim_pos * torch.exp(-1.0 * sim_neg.clamp(min=0.0) / sigma)
        # decay_term = sim_neg.clamp(min=0.0) / pos_neg_sim_selected.clamp(min=1e-10)
        # sim_final = sim_pos * torch.exp(-1.0 * decay_term / sigma)
        return sim_final

    def _compute_sim_instance_softmax(self, tar_feat, masks_feat_size_bool, temp=1.0):
        n_masks = masks_feat_size_bool.shape[0]
        c = tar_feat.shape[-1]

        tar_avg_feats_all = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)  # [n_mask, c]
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_ins_feats_avg = F.normalize(self.mem_feats_ins_avg.flatten(0, 1), p=2, dim=-1)

        sim_avg = tar_avg_feats @ mem_ins_feats_avg.t()
        sim_avg = sim_avg.reshape(n_masks, self.mem_n_classes, self.mem_length)

        scores = torch.softmax((sim_avg / temp).sum(dim=-1), dim=-1)
        # scores = torch.softmax(sim_avg / temp, dim=-1).reshape(n_masks, self.mem_n_classes, self.mem_length)
        # scores = scores.sum(dim=-1)
        return scores

    def _compute_sim_center(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        bboxes = batched_mask_to_box(masks_feat_size_bool.reshape(n_masks, self.encoder_h, self.encoder_w))
        x_centers = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
        y_centers = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        x_centers = (x_centers / self.encoder_w - 0.5) * 2.0
        y_centers = (y_centers / self.encoder_h - 0.5) * 2.0
        xy_centers = torch.stack((x_centers, y_centers), dim=-1).reshape(1, 1, n_masks, 2)

        tar_feat_2d = tar_feat.reshape(1, self.encoder_h, self.encoder_w, self.encoder_dim).permute(0, 3, 1, 2)
        sampled_feats = F.grid_sample(tar_feat_2d, xy_centers, mode='bilinear').reshape(self.encoder_dim, n_masks).t()
        sampled_feats = F.normalize(sampled_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg  # [n_class, c]
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        sim_avg = sampled_feats @ mem_feats_avg.t()  # [n_masks, n_class]
        sim_avg = (sim_avg + 1.0) * 0.5
        return sim_avg

    def _compute_sim_matching(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        tar_avg_feats_all = []
        tar_foreground_feats = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))
            tar_foreground_feats.append(feats_i)

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)

        mem_feats_avg = self.mem_feats_avg
        mem_feats_avg = F.normalize(mem_feats_avg, p=2, dim=-1)
        mem_feats_centers = self.mem_feats_centers  # already normed
        mem_feats_centers = mem_feats_centers.reshape(-1, self.encoder_dim)  # [n_class * n_centers, c]

        sim_global = tar_avg_feats @ mem_feats_avg.t()
        sim_global = (sim_global + 1.0) * 0.5

        sims_matching = []
        for i in range(n_masks):
            tar_feats_i = tar_foreground_feats[i]  # [n_fore, c]
            _norm = float(tar_feats_i.shape[0]) * 10.0  # to avoid over float

            tar_mag_i = torch.sqrt(torch.pow((tar_feats_i / _norm).sum(dim=0), 2).sum())  # [1,]

            sim = tar_feats_i @ mem_feats_centers.t()  # [n_fore, n_classes * n_centers]
            sim = sim.reshape(-1, self.mem_n_classes, self.kmeans_k)
            sim = sim.max(dim=-1)[0]  # [n_fore, n_classes]
            sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            sim = sim / tar_mag_i
            sim = (sim + 1.0) * 0.5

            # sim = tar_feats_i @ mem_feats_avg.t()   # [n_fore, n_classes]
            # sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            # sim = sim / tar_mag_i
            # sim = (sim + 1.0) * 0.5

            sims_matching.append(sim)
        sim_matching = torch.cat(sims_matching, dim=0)
        r = 0.0
        similarity = sim_global * r + sim_matching * (1.0 - r)
        return similarity

    def _compute_sim_knn(self, tar_feat, masks_feat_size_bool, k=5, sigma=0.01):
        device = tar_feat.device
        n_masks = masks_feat_size_bool.shape[0]

        tar_avg_feats_all = []

        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))

        tar_avg_feats = torch.cat(tar_avg_feats_all, dim=0)
        tar_avg_feats = F.normalize(tar_avg_feats, p=2, dim=-1)
        #
        # mem_ins_feats_avg = (
        #     torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=2)
        #     / self.mem_masks.sum(dim=2).unsqueeze(dim=2)
        # )
        mem_ins_feats_avg = F.normalize(self.mem_feats_ins_avg.flatten(0, 1), p=2, dim=-1)

        sim = tar_avg_feats @ mem_ins_feats_avg.t()  # [n_mask, n_class * n_ins]
        dis = 1.0 - sim
        sim = (sim + 1.0) * 0.5

        # sim = sim.reshape(n_masks, self.mem_n_classes, self.mem_length).mean(dim=-1)
        # scores, labels = torch.topk(sim, 1)
        # scores = scores.flatten()
        # labels = labels.flatten()

        top_sim, top_inds = torch.topk(sim, k=k, dim=-1)   # [n_mask, k]
        top_class_inds = top_inds // self.mem_length  # [n_mask, k]
        if k == 1:
            return top_sim.flatten(), top_class_inds.flatten()

        labels = torch.zeros((n_masks), dtype=torch.long, device=device)
        scores = torch.zeros((n_masks), dtype=torch.float32, device=device)
        for i in range(n_masks):
            counts = torch.bincount(top_class_inds[i], minlength=self.mem_n_classes)
            _label = torch.argmax(counts)
            _hit_inds = top_class_inds[i] == _label
            # _score = top_sim[i][top_class_inds[i] == _label].mean()
            ws = torch.exp(-1 * torch.pow(dis[i][top_inds[i]][_hit_inds], 2) / sigma**2)
            _score = (top_sim[i][_hit_inds] * ws).sum() / ws.sum()
            labels[i] += _label
            scores[i] += _score

        return scores, labels

    def _compute_sim_covariance_cosine(self, tar_feat, masks_feat_size_bool):
        device = tar_feat.device

        n_masks = masks_feat_size_bool.shape[0]
        n_classes = self.mem_n_classes
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]
        x = x.unsqueeze(dim=1)  # [n_masks, 1, c]

        t = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, c]

        sigma = self.mem_feats_covariances
        lamda_s = 0.001
        sigma = (1.0 - lamda_s) * sigma + lamda_s * torch.eye(c, device=device).unsqueeze(dim=0)

        invert_sigma = torch.linalg.inv(sigma)  # [n_class, c, c]

        sim_classes = []
        for i in range(n_classes):
            x_norm = torch.sqrt(x @ invert_sigma[i:i+1] @ x.transpose(1, 2)).reshape(n_masks, 1)
            t_norm = torch.sqrt(t[i:i+1] @ invert_sigma[i] @ t[i:i+1].t()).reshape(1)
            sim = (x @ invert_sigma[i:i+1] @ t[i].reshape(1, c, 1)).reshape(n_masks, 1)
            sim = sim / (x_norm * t_norm)
            sim_classes.append(sim)
        scores = torch.cat(sim_classes, dim=1)
        scores = scores.clamp(min=0.0)
        return scores

    def _compute_sim_intra_class_norm(self, tar_feat, masks_feat_size_bool, margin=0.1, sigma=1.0):
        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]
        x = x.unsqueeze(dim=1)  # [n_masks, c]
        x_norm = F.normalize(x, p=2, dim=-1)

        t_ins = self.mem_feats_ins_avg
        t_ins_norm = F.normalize(t_ins, p=2, dim=-1)
        intra_class_sim = t_ins_norm @ t_ins_norm.transpose(-1, -2)
        intra_class_sim = intra_class_sim.mean(dim=(1, 2))
        intra_class_sim = intra_class_sim.unsqueeze(dim=0)

        t = t_ins.mean(dim=1)  # [n_class, c]
        t_norm = F.normalize(t, p=2, dim=-1)

        sim = x_norm @ t_norm.t()
        sim = sim.clamp(min=0.0)
        sim = torch.where(
            sim > intra_class_sim - margin,
            sim,
            sim / torch.pow(intra_class_sim, sigma)
        )
        return sim

    def _compute_sim_covariance_diag_cosine(self, tar_feat, masks_feat_size_bool):
        device = tar_feat.device

        n_masks = masks_feat_size_bool.shape[0]
        n_classes = self.mem_n_classes
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]

        # t = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, c]
        t = self.mem_feats_avg

        _inds = torch.arange(c, device=device)
        diag_cov = self.mem_feats_covariances[:, _inds, _inds]  # [n_class, c]
        t_scaled = t / torch.sqrt(diag_cov)
        t_scaled = F.normalize(t_scaled, p=2, dim=-1)

        sim_classes = []
        for i in range(n_classes):
            x_scaled = x / diag_cov[i:i+1]
            x_scaled = F.normalize(x_scaled, p=2, dim=-1)
            sim = x_scaled @ t_scaled[i:i+1].t()
            sim_classes.append(sim)

        scores = torch.cat(sim_classes, dim=1)
        scores = scores.clamp(min=0.0)
        return scores

    def _compute_sim_vMF(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]
        n_classes = self.mem_n_classes
        c = tar_feat.shape[-1]

        masks = masks_feat_size_bool.to(dtype=tar_feat.dtype)
        x = (masks @ tar_feat) / masks.sum(dim=-1, keepdim=True)  # [n_masks, c]
        mu = self.mem_feats_ins_avg.mean(dim=1)  # [n_class, c]
        r = self.mem_feats_ins_avg.sum(dim=1).norm(p=2, dim=-1) / self.mem_feats_ins_avg.shape[1]
        kappa = (r * (c - r ** 2)) / (1 - r ** 2)

    def _compute_sim_matching_soft(self, tar_feat, masks_feat_size_bool):
        n_masks = masks_feat_size_bool.shape[0]

        tar_avg_feats_all = []
        tar_foreground_feats = []
        for i in range(n_masks):
            feats_i = tar_feat[masks_feat_size_bool[i]]
            tar_avg_feats_all.append(feats_i.mean(dim=0, keepdim=True))
            tar_foreground_feats.append(feats_i)

        mem_feats_centers = self.mem_feats_centers  # already normed
        mem_feats_centers = mem_feats_centers.reshape(-1, self.encoder_dim)  # [n_class * n_centers, c]

        sims_matching = []
        for i in range(n_masks):
            tar_feats_i = tar_foreground_feats[i]  # [n_fore, c]
            _norm = float(tar_feats_i.shape[0]) * 10.0  # to avoid over float

            tar_mag_i = torch.sqrt(torch.pow((tar_feats_i / _norm).sum(dim=0), 2).sum())  # [1,]

            sim = tar_feats_i @ mem_feats_centers.t()  # [n_fore, n_classes * n_centers]
            sim = sim.reshape(-1, self.mem_n_classes, self.kmeans_k)

            theta = 1.0
            w = torch.exp(sim/theta)
            sim = (sim * w).sum(dim=-1) / w.sum(dim=-1)

            # sim = sim.max(dim=-1)[0]  # [n_fore, n_classes]
            sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            sim = sim / tar_mag_i
            sim = (sim + 1.0) * 0.5

            # sim = tar_feats_i @ mem_feats_avg.t()   # [n_fore, n_classes]
            # sim = (sim / _norm).sum(dim=0, keepdim=True)  # [1, n_classes]
            # sim = sim / tar_mag_i
            # sim = (sim + 1.0) * 0.5

            sims_matching.append(sim)
        sim_matching = torch.cat(sims_matching, dim=0)
        return sim_matching

    def forward_test(self, input_dicts, with_negative):
        assert len(input_dicts) == 1

        device = self.predictor.device

        tar_img = input_dicts[0]["target_img"].to(device=device)
        sam_input_size = tar_img.shape[-2]
        tar_img_encoder = F.interpolate(
            tar_img.unsqueeze(dim=0),
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        tar_feat, last_attn = self._forward_encoder_attn_roll(self.encoder_transform(tar_img_encoder))
        tar_feat = tar_feat.reshape(-1, self.encoder_dim)  # [N, C]

        # ----------------------------------------------------------------------------------------
        # SAM inference
        tar_img = tar_img.unsqueeze(dim=0)

        # Method 1: first matching then SAM
        # match_size = (37, 37)
        # num_points = min(match_size[0] * match_size[1], 1024)
        # precomputed_points = self._compute_query_points(tar_feat, match_size, num_points)
        # lr_masks, pred_ious, query_points = self._forward_sam(
        #     self.sam_transform(tar_img), precomputed_points, point_normed=True
        # )

        # Method 2: Normal inference
        lr_masks, pred_ious, query_points = self._forward_sam(self.sam_transform(tar_img))

        # Method 3: Multi-scale inference
        # lr_masks, pred_ious, query_points = self._forward_sam_multiscale(tar_img, scales=(0.7, 0.8, 0.9, 1.0))
        # ----------------------------------------------------------------------------------------

        n_masks = lr_masks.shape[0]
        masks_feat_size_bool = lr_masks > 0
        masks_feat_size_bool = masks_feat_size_bool.reshape(n_masks, -1)
        tar_feat = tar_feat.reshape(1, self.encoder_h, self.encoder_w, -1).permute(0, 3, 1, 2)
        tar_feat = F.interpolate(
            tar_feat,
            size=tuple(lr_masks.shape[-2:]),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).reshape(-1, lr_masks.shape[-2] * lr_masks.shape[-1]).t()

        if not with_negative:
            # sim_local = self._compute_sim_matching(tar_feat, masks_feat_size_bool)
            # pca_scores = self._compute_pca_scores(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_center(tar_feat, masks_feat_size_bool)
            sim_global, obj_feats = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool, ret_feats=True)
            # sim_global = self._compute_sim_covariance_diag_cosine(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_vMF(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_intra_class_norm(tar_feat, masks_feat_size_bool, margin=0.2, sigma=0.2)
            # sim_global = self._compute_sim_covariance_cosine(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool, softmax=True, temp=0.5)
            # sim_global = self._compute_sim_instance_softmax(tar_feat, masks_feat_size_bool, temp=0.75)
            # sim_global = self._compute_sim_attn_guided_global_avg(tar_feat, last_attn, masks_feat_size_bool)
        else:
            assert self.with_negative_refs
            assert self.memory_neg_ready
            # sim_global = self._compute_sim_attn_guided_global_weighted(tar_feat, last_attn, masks_feat_size_bool)
            # sim_global = self._compute_sim_attn_guided_global_avg(tar_feat, last_attn, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool)
            sim_global = self._compute_sim_global_avg_with_neg(tar_feat, masks_feat_size_bool, sigma=0.8)
            # sim_global = self._compute_sim_global_l2(tar_feat, masks_feat_size_bool, sigma=30.0)
            # sim_global = self._compute_sim_covariance_cosine(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_gaussian(tar_feat, masks_feat_size_bool)

        merged_scores = sim_global  #  * torch.pow(pca_scores, 0.25)

        if self.cls_num_per_mask == -1:
            self.cls_num_per_mask = self.mem_n_classes
        top_scores, labels = torch.topk(merged_scores, k=self.cls_num_per_mask)

        if self.cls_num_per_mask == self.mem_n_classes:
            max_scores = top_scores[:, 0:1]
            top_scores = top_scores * (top_scores > (max_scores * 0.6))

        labels = labels.flatten()
        scores_all_class = top_scores.flatten()

        # scores_thr_by_classes = self.mem_ins_sim_avg[labels] * 0.8
        # scores_all_class[scores_all_class < scores_thr_by_classes] = 0.0

        # scores_all_class, labels = self._compute_sim_knn(tar_feat, masks_feat_size_bool, k=7, sigma=0.1)

        # assert self.cls_num_per_mask == 1
        # ambiguous_decay = self._compute_ambiguous_decay(sim_global, labels)
        # scores_all_class = scores_all_class * ambiguous_decay

        # ----------------------------------------------------------------------------------------
        # Local-global similarity analysis
        # n_masks = masks_feat_size_bool.shape[0]
        # local_global_mean = []
        # local_global_std = []
        # for i in range(n_masks):
        #     _feats_fore = tar_feat[masks_feat_size_bool[i]]
        #     _feats_avg = _feats_fore.mean(dim=0, keepdim=True)
        #     _sim = F.normalize(_feats_fore, p=2, dim=-1) @ F.normalize(_feats_avg, p=2, dim=-1).t()
        #     _sim = (_sim + 1.0) * 0.5
        #     local_global_std.append(torch.std(_sim))
        #     local_global_mean.append(torch.mean(_sim))
        # local_global_mean = torch.stack(local_global_mean).flatten()
        # local_global_std = torch.stack(local_global_std).flatten()
        # ----------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------
        # Oracle Analysis
        # scores_oracle = self._get_oracle_iou(lr_masks, input_dicts[0]["tar_anns_by_cat"]).t()
        # assert self.cls_num_per_mask == 1
        #
        # top_scores, labels = torch.topk(scores_oracle, k=self.cls_num_per_mask)
        # labels = labels.flatten()
        # scores_all_class = top_scores.flatten()

        #
        # select_sim_global = sim_global[torch.arange(scores_oracle.shape[0], device=device), labels].flatten()
        # select_scores_oracle = scores_oracle[torch.arange(scores_oracle.shape[0], device=device), labels].flatten()
        # select_labels = labels.flatten()
        # selected_ins_sims = self.mem_ins_sim_avg[labels].flatten()
        # select_pca_scores = pca_scores[torch.arange(scores_oracle.shape[0], device=device), labels].flatten()

        # top_scores_oracle, oracle_labels = torch.topk(scores_oracle, k=self.cls_num_per_mask)
        # oracle_labels = oracle_labels.flatten()
        # top_scores_oracle = top_scores_oracle.flatten()
        # ----------------------------------------------------------------------------------------

        lr_bboxes = batched_mask_to_box(lr_masks > 0)
        lr_bboxes_expand = (
            lr_bboxes.unsqueeze(dim=1)
            .expand(-1, self.cls_num_per_mask, -1)
            .reshape(n_masks * self.cls_num_per_mask, 4)
        )

        expand_ratio = 8
        out_num = int(min(self.num_out_instance * expand_ratio, labels.shape[0]))

        nms_keep_inds = batched_nms(
            lr_bboxes_expand.float(),
            # scores_all_class,
            pred_ious.flatten(),
            labels,
            iou_threshold=self.nms_thr
        )[:out_num]
        scores_out = scores_all_class[nms_keep_inds]
        pred_ious_out = pred_ious[nms_keep_inds]
        lr_masks_out = lr_masks[nms_keep_inds // self.cls_num_per_mask]
        obj_feats_out = obj_feats[nms_keep_inds // self.cls_num_per_mask]
        masks_feat_size_bool = masks_feat_size_bool[nms_keep_inds // self.cls_num_per_mask]
        labels_out = labels[nms_keep_inds]

        pos_inds = scores_out > 0.0
        scores_out = scores_out[pos_inds]
        lr_masks_out = lr_masks_out[pos_inds]
        obj_feats_out = obj_feats_out[pos_inds]
        masks_feat_size_bool = masks_feat_size_bool[pos_inds]
        labels_out = labels_out[pos_inds]

        # query_points = query_points[keep_inds // self.cls_num_per_mask]

        # ----------------------------------------------------------------------------------------
        # Iteratively Refine
        # refine_points = query_points.reshape(self.num_out_instance, 1, 2)
        # refine_point_labels = torch.ones_like(refine_points[:, :, 0], dtype=torch.int32).reshape(-1, 1)

        # refine_points, refine_labels, do_refine = self._get_oracle_refine_prompts(
        #     lr_masks_out, labels_out, input_dicts[0]["tar_anns_by_cat"], pool_size=5
        # )

        # refine_points, refine_labels = self._get_refine_prompts(
        #     tar_feat, lr_masks_out, labels_out, pool_size=5, thr=0.6
        # )
        # do_refine = torch.ones((lr_masks_out.shape[0],), device=device)
        #
        # do_refine = do_refine > 0
        # if do_refine.sum() > 0:
        #     refine_point_inputs = dict(
        #         point_coords=refine_points[do_refine],
        #         point_labels=refine_labels[do_refine].to(dtype=torch.int32)
        #     )
        #     lr_masks_refine, _ = self._compute_masks_refine(
        #         point_inputs=refine_point_inputs, boxes_inputs=None, mask_inputs=lr_masks_out[do_refine]
        #     )
        #     lr_masks_out[do_refine] = lr_masks_refine
        # lr_masks_out = lr_masks_out.reshape(self.num_out_instance, *lr_masks_out.shape[-2:])
        # ----------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------
        # Other decay

        # score_decay = 1.0 - self._compute_ios(lr_masks_out>0, labels_out, rank_score=True)
        # scores_out = scores_out * torch.pow(score_decay, 0.1)

        # compelteness = self._compute_completeness_decay(tar_feat, masks_feat_size_bool, scores_out, labels_out, decay=0.4)
        # scores_out = scores_out * torch.pow(compelteness, 0.2)

        # unificationess = self._compute_unification_decay(tar_feat, masks_feat_size_bool)
        # scores_out = scores_out * torch.pow(unificationess, 0.1)

        # sim_neg = self._compute_negative_decay(tar_feat, masks_feat_size_bool, scores_out, labels_out)
        # s_square = 1.0
        # scores_out = scores_out * torch.exp(-1.0 * (sim_neg - scores_out).clamp(min=0.0) / s_square)
        # ----------------------------------------------------------------------------------------

        # resizing and converting to output format
        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]

        
        if lr_masks_out.shape[0] == 0:
            raise ValueError("No masks found")
            # self._reset()
            # return [{
            #     "binary_masks": torch.zeros((0, ori_h, ori_w), device=device).bool(),
            #     "bboxes": torch.zeros((0, 4), device=device),
            #     "scores": torch.zeros((0,), device=device),
            #     "labels": torch.zeros((0,), dtype=torch.long, device=device),
            #     "image_info": input_dicts[0]["target_img_info"],
            # }]

        masks_out_binary = F.interpolate(
            lr_masks_out.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_out_binary)

        # ----------------------------------------------------------------------------------------
        # Merging Masks

        # use_batches = False # This should be False for better results
        # if use_batches:
        #     batch_size = 10
        #     n_masks = masks_out_binary.shape[0]
        #     all_score_decays = []
        #     for i in range(0, n_masks, batch_size):
        #         batch_end = min(i + batch_size, n_masks)
        #         batch_masks = masks_out_binary[i:batch_end]
        #         batch_labels = labels_out[i:batch_end]
        #         batch_score_decay = 1.0 - self._compute_ios_batched(
        #             batch_masks, 
        #             batch_labels,
        #             rank_score=True,
        #             batch_size=batch_size
        #         )
        #         all_score_decays.append(batch_score_decay)
        #     score_decay = torch.cat(all_score_decays)
        # else:
        #     obj_sim = obj_feats_out @ obj_feats_out.t()
        #     obj_sim = obj_sim.clamp(min=0.0)
        #     ios = self._compute_semantic_ios(masks_out_binary, labels_out, obj_sim, use_semantic=True, rank_score=True)
        #     score_decay = 1 - ios
        #     # # # # # Old version
        #     # # # # score_decay = 1.0 - self._compute_ios(masks_out_binary, labels_out, rank_score=True)


        # Method 1: Soft merging
        obj_sim = obj_feats_out @ obj_feats_out.t()
        obj_sim = obj_sim.clamp(min=0.0)
        ios = self._compute_semantic_ios(masks_out_binary, labels_out, obj_sim, use_semantic=True, rank_score=True)
        score_decay = 1 - ios
        scores_out = scores_out * torch.pow(score_decay, 0.5)

        # Method 2: Hard merging
        # ios = self._compute_ios(masks_out_binary, labels_out, rank_score=False)
        # keep_inds = ios < 1.0

        # scores_out = scores_out[keep_inds]
        # masks_out_binary = masks_out_binary[keep_inds]
        # bboxes = bboxes[keep_inds]
        # labels_out = labels_out[keep_inds]
        # ----------------------------------------------------------------------------------------

        final_out_num = min(self.num_out_instance, scores_out.shape[0])
        final_out_inds = torch.argsort(scores_out, descending=True)[:final_out_num]

        pred_ious_out = pred_ious_out[final_out_inds]
        masks_out_binary = masks_out_binary[final_out_inds]
        bboxes = bboxes[final_out_inds]
        scores_out = scores_out[final_out_inds]
        labels_out = labels_out[final_out_inds]

        # score_to_analysis = torch.stack(
        #     (
        #         select_sim_global[nms_keep_inds][pos_inds][final_out_inds],
        #         select_labels[nms_keep_inds][pos_inds][final_out_inds],
        #         select_scores_oracle[nms_keep_inds][pos_inds][final_out_inds]
        #     ),
        #     dim=-1
        # )

        output_dict = dict(
            binary_masks=masks_out_binary,
            bboxes=bboxes,
            scores=scores_out,  # select_scores_oracle[nms_keep_inds][pos_inds][final_out_inds], #, #
            labels=labels_out,
            # score_to_analysis=score_to_analysis,
            image_info=input_dicts[0]["target_img_info"],
        )

        # self._vis_results_online(output_dict, input_dicts[0]["tar_anns_by_cat"], score_thr=0.5, show_scores=True, dataset_name=self.dataset_name)
        self._vis_results_online(output_dict, input_dicts[0]["tar_anns_by_cat"], score_thr=0.5, show_scores=True, dataset_name='lvis')
        self._reset()
        return [output_dict]

    def testing_classifier(self, input_dicts, with_negative):
        assert len(input_dicts) == 1

        device = self.predictor.device

        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]

        tar_img = input_dicts[0]["target_img"].to(device=device)
        sam_input_size = tar_img.shape[-2]
        tar_img_encoder = F.interpolate(
            tar_img.unsqueeze(dim=0),
            size=(self.encoder_img_size, self.encoder_img_size),
            mode="bicubic"
        )
        tar_feat, last_attn = self._forward_encoder_attn_roll(self.encoder_transform(tar_img_encoder))
        tar_feat = tar_feat.reshape(-1, self.encoder_dim)  # [N, C]

        tar_anns_by_cat = input_dicts[0]["tar_anns_by_cat"]

        masks = []
        for cat_ind in tar_anns_by_cat.keys():
            masks.append(tar_anns_by_cat[cat_ind]["masks"].to(device=device))

        if len(masks) > 0:
            masks = torch.cat(masks, dim=0)
        else:
            masks = torch.ones((1, sam_input_size, sam_input_size)).to(device=device, dtype=torch.float32)

        n_masks = masks.shape[0]
        masks_feat_size = F.interpolate(
            masks.unsqueeze(dim=1),
            size=(self.encoder_h, self.encoder_w),
            mode="nearest"
        ).reshape(n_masks, -1)

        masks_feat_size_bool = masks_feat_size > 0
        non_empty_inds = masks_feat_size_bool.sum(dim=1) > 0

        masks_feat_size = masks_feat_size[non_empty_inds]
        masks_feat_size_bool = masks_feat_size_bool[non_empty_inds]
        masks_out = masks[non_empty_inds]

        n_masks = masks_feat_size_bool.shape[0]
        if n_masks == 0:
            output_dict = dict(
                binary_masks=torch.zeros((0, ori_h, ori_w)) > 0,
                bboxes=torch.zeros((0, 4)),
                scores=torch.zeros((0,)),
                labels=torch.zeros((0,)),
                # score_to_analysis=score_to_analysis,
                image_info=input_dicts[0]["target_img_info"],
            )
            return [output_dict]

        if not with_negative:
            # sim_local = self._compute_sim_matching(tar_feat, masks_feat_size_bool)
            # pca_scores = self._compute_pca_scores(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_center(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool, softmax=True, temp=0.5)
            # sim_global = self._compute_sim_instance_softmax(tar_feat, masks_feat_size_bool, temp=0.75)
            sim_global = self._compute_sim_attn_guided_global_avg(tar_feat, last_attn, masks_feat_size_bool)
        else:
            assert self.with_negative_refs
            assert self.memory_neg_ready
            sim_global = self._compute_sim_global_avg(tar_feat, masks_feat_size_bool)
            # sim_global = self._compute_sim_global_avg_with_neg(tar_feat, masks_feat_size_bool, margin=0.6, sigma=1.0)

        self.cls_num_per_mask = 1
        merged_scores = sim_global  # * torch.pow(pca_scores, 0.25)
        top_scores, labels = torch.topk(merged_scores, k=self.cls_num_per_mask)

        # top_scores, labels = self._compute_sim_knn(tar_feat, masks_feat_size_bool, k=1, sigma=0.1)

        labels_out = labels.flatten()
        scores_out = top_scores.flatten()

        masks_out_binary = F.interpolate(
            masks_out.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="bilinear",
            align_corners=False,
            antialias=True
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_out_binary)

        final_out_num = min(self.num_out_instance, scores_out.shape[0])
        final_out_inds = torch.argsort(scores_out, descending=True)[:final_out_num]
        masks_out_binary = masks_out_binary[final_out_inds]
        bboxes = bboxes[final_out_inds]
        scores_out = scores_out[final_out_inds]
        labels_out = labels_out[final_out_inds]

        # score_to_analysis = torch.stack((local_global_mean, local_global_std, select_scores_oracle), dim=-1)

        output_dict = dict(
            binary_masks=masks_out_binary,
            bboxes=bboxes,
            scores=scores_out,
            labels=labels_out,
            # score_to_analysis=score_to_analysis,
            image_info=input_dicts[0]["target_img_info"],
        )
        # self._vis_results_online(output_dict, input_dicts[0]["tar_anns_by_cat"], score_thr=0.0, show_scores=True)
        return [output_dict]

    def postprocess_memory(self):
        # Compute class-wise average features
        device = self.mem_feats_avg.device
        c = self.mem_feats.shape[-1]

        self.mem_feats_avg *= 0.0
        mem_feats_avg = (
            torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=(1, 2))
            / self.mem_masks.sum(dim=(1, 2)).unsqueeze(dim=1)
        )
        self.mem_feats_avg += mem_feats_avg

        mem_feats_ins_avg = (
            torch.sum(self.mem_feats * self.mem_masks.unsqueeze(dim=-1), dim=2)
            / self.mem_masks.sum(dim=2).unsqueeze(dim=2)
        )
        self.mem_feats_ins_avg += mem_feats_ins_avg

        sigmas = []
        for i in range(self.mem_n_classes):
            # feats_i = self.mem_feats[i].reshape(-1, c)
            # masks_i = self.mem_masks[i].reshape(-1)
            # feats_i_fore = feats_i[masks_i > 0]
            # mu_i = feats_i_fore.mean(dim=0, keepdim=True)
            # feats_i_centered = feats_i_fore - mu_i

            feats_i = self.mem_feats_ins_avg[i].reshape(-1, c)
            mu_i = feats_i.mean(dim=0, keepdim=True)
            feats_i_centered = feats_i - mu_i

            sigma_i = feats_i_centered.t() @ feats_i_centered / float(feats_i_centered.shape[0])
            sigmas.append(sigma_i.unsqueeze(dim=0))
        sigmas = torch.cat(sigmas, dim=0)
        self.mem_feats_covariances += sigmas




        # compute mean sim, method 1
        # ins_sims = []
        # for i in range(self.mem_n_classes):
        #     feats_i = []
        #     for j in range(self.mem_length):
        #         feat_ij = self.mem_feats[i, j]
        #         mask_ij = self.mem_masks[i, j]
        #         feats_i.append(feat_ij[mask_ij > 0].mean(dim=0, keepdim=True))
        #     feats_i = F.normalize(torch.cat(feats_i, dim=0), p=2, dim=-1)
        #     ins_sims_i = feats_i @ feats_i.t()
        #     ins_sims_i = (ins_sims_i + 1.0) * 0.5
        #     ins_sims_i = ins_sims_i.flatten()
        #     ins_sims_i = ins_sims_i[ins_sims_i < 1.0].mean()
        #     ins_sims.append(ins_sims_i)
        # ins_sims = torch.stack(ins_sims).reshape(self.mem_n_classes)
        # self.mem_ins_sim_avg += ins_sims

        # compute mean sim, method 2
        ins_sims = []
        for i in range(self.mem_n_classes):
            sims_i = []
            for j in range(self.mem_length):
                feat_ij = self.mem_feats[i, j]
                feats_i_rest = torch.cat((self.mem_feats[i, :j], self.mem_feats[i, j+1:]), dim=0)
                mask_ij = self.mem_masks[i, j]
                mask_i_rest = torch.cat((self.mem_masks[i, :j], self.mem_masks[i, j+1:]), dim=0)

                feats_ij = F.normalize(feat_ij[mask_ij > 0].mean(dim=0, keepdim=True), p=2, dim=1)
                feats_i_rest = F.normalize(feats_i_rest[mask_i_rest > 0].mean(dim=0, keepdim=True), p=2, dim=-1)
                sim_ij = feats_ij @ feats_i_rest.t()
                sim_ij = (sim_ij + 1.0) * 0.5
                sims_i.append(sim_ij)
            sim_i = torch.stack(sims_i).mean()
            ins_sims.append(sim_i)
        ins_sims = torch.stack(ins_sims).reshape(self.mem_n_classes)
        self.mem_ins_sim_avg += ins_sims

        # K-means
        kmeans_iters = 100
        for i in range(self.mem_n_classes):
            feats = self.mem_feats[i].reshape(-1, self.encoder_dim)[self.mem_masks[i].reshape(-1) > 0]
            assert feats.shape[0] > 0
            centers_i = kmeans(feats, self.kmeans_k, kmeans_iters)
            # centers_i = kmeans_decouple(feats, feats_fore, self.kmeans_k, kmeans_iters)
            self.mem_feats_centers[i] += centers_i

        # PCA
        for i in range(self.mem_n_classes):
            feats = self.mem_feats[i].reshape(-1, self.encoder_dim)[self.mem_masks[i].reshape(-1) > 0]
            assert feats.shape[0] > 0
            feats = feats.cpu().numpy()
            pca = PCA(n_components=self.n_pca_components)
            pca.fit(feats)
            pca_mean = torch.from_numpy(pca.mean_).to(device=device)
            pca_components = torch.from_numpy(pca.components_).to(device=device)
            self.mem_pca_mean[i] += pca_mean
            self.mem_pca_components[i] += pca_components

        # TODO: FoundPose's Method


        self.mem_postprocessed[0] = True

    def postprocess_memory_negative(self):
        device = self.mem_feats_avg.device

        mem_feats_avg_neg = (
                torch.sum(self.mem_feats_neg * self.mem_masks_neg.unsqueeze(dim=-1), dim=(1, 2))
                / self.mem_masks_neg.sum(dim=(1, 2)).unsqueeze(dim=1)
        )
        self.mem_feats_avg_neg += mem_feats_avg_neg

        mem_feats_ins_avg_neg = (
                torch.sum(self.mem_feats_neg * self.mem_masks_neg.unsqueeze(dim=-1), dim=2)
                / self.mem_masks_neg.sum(dim=2).unsqueeze(dim=2)
        )
        self.mem_feats_ins_avg_neg += mem_feats_ins_avg_neg
        self.mem_postprocessed_neg[0] = True

    def forward(self, input_dicts):
        data_mode = input_dicts[0].pop("data_mode", None)

        assert data_mode is not None
        assert not self.training

        if data_mode == "fill_memory":
            return self.forward_fill_memory(input_dicts, is_positive=True)
        elif data_mode == "fill_memory_neg":
            assert self.with_negative_refs
            assert not self.memory_neg_ready
            assert not self.mem_postprocessed_neg[0].item()
            return self.forward_fill_memory(input_dicts, is_positive=False)
        elif data_mode == "vis_memory":
            return self.forward_vis_memory(input_dicts)
        elif data_mode == "test":
            if self.with_negative_refs:
                if not self.memory_ready:
                    if self.mem_postprocessed[0].item():
                        self.memory_ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                if not self.memory_neg_ready:
                    if self.mem_postprocessed_neg[0].item():
                        self.memory_neg_ready = True
                    else:
                        raise RuntimeError("Negative memory is not ready!")

                return self.forward_test(input_dicts, with_negative=True)
                # return self.testing_classifier(input_dicts, with_negative=True)
            else:
                if not self.memory_ready:
                    if self.mem_postprocessed[0].item():
                        self.memory_ready = True
                    else:
                        raise RuntimeError("Memory is not ready!")
                return self.forward_test(input_dicts, with_negative=False)
                # return self.testing_classifier(input_dicts, with_negative=False)
        elif data_mode == "test_support":
            assert self.with_negative_refs
            if not self.memory_ready:
                if self.mem_postprocessed[0].item():
                    self.memory_ready = True
                else:
                    raise RuntimeError("Memory is not ready!")
            assert not self.memory_neg_ready
            assert not self.mem_postprocessed_neg[0].item()
            return self.forward_test(input_dicts, with_negative=False)
            # return self.testing_classifier(input_dicts, with_negative=False)
        else:
            raise NotImplementedError(f"Unrecognized data mode during inference: {data_mode}")

    def _vis_results_online(self, output_dict, tar_anns_by_cat, score_thr=0.65, show_scores=False, dataset_name=None):
        import os
        from dev_hongyi.dataset.visualization import vis_coco

        scores = output_dict["scores"].cpu().numpy()
        masks_pred = output_dict["binary_masks"].cpu().numpy()
        bboxes = output_dict["bboxes"].cpu().numpy()
        labels = output_dict["labels"].cpu().numpy()

        image_info = output_dict["image_info"]
        if dataset_name == "coco":
            img_path = os.path.join(f"./data/{dataset_name}/val2017", image_info["file_name"])
        elif dataset_name == "lvis":
            img_path = os.path.join(f"./data/coco/allimages", image_info["file_name"])
        else:
            img_path = os.path.join(f"./data/{dataset_name}/test", image_info["file_name"])
        out_path = os.path.join(f"./results_analysis/comparison/{dataset_name}", image_info["file_name"])

        gt_masks = []
        gt_bboxes = []
        gt_labels = []

        for cat_ind in tar_anns_by_cat.keys():
            gt_masks.append(tar_anns_by_cat[cat_ind]["masks"].cpu().numpy())
            gt_bboxes.append(tar_anns_by_cat[cat_ind]["bboxes"].cpu().numpy())
            gt_labels.extend([cat_ind for _ in range(len(tar_anns_by_cat[cat_ind]["masks"]))])
        if len(gt_bboxes) > 0:
            gt_bboxes = np.concatenate(gt_bboxes)
            gt_masks = np.concatenate(gt_masks)

            gt_bboxes[:, 0] = gt_bboxes[:, 0] * image_info["ori_width"] / self.sam_img_size
            gt_bboxes[:, 1] = gt_bboxes[:, 1] * image_info["ori_height"] / self.sam_img_size
            gt_bboxes[:, 2] = gt_bboxes[:, 2] * image_info["ori_width"] / self.sam_img_size
            gt_bboxes[:, 3] = gt_bboxes[:, 3] * image_info["ori_height"] / self.sam_img_size

            # TODO: resize gt masks when I am not lazy lol

        vis_coco(
            gt_bboxes,
            gt_labels,
            gt_masks,
            scores,
            labels,
            bboxes,
            masks_pred,
            score_thr=score_thr,
            img_path=img_path,
            out_path=out_path,
            show_scores=show_scores,
            dataset_name=dataset_name
        )

