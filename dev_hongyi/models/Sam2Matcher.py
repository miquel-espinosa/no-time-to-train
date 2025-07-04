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
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import fill_holes_in_mask_scores, concat_points
from sam2.utils.amg import batched_mask_to_box

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import dinov2.dinov2.models.vision_transformer as dinov2_vit
import dinov2.dinov2.utils.utils as dinov2_utils

from dev_hongyi.models.matcher_utils import kmeans_pp, SAM2AutomaticMaskGenerator_Matcher
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

class RobustPromptSampler:

    def __init__(
        self,
        encoder_feat_size,
        sample_range,
        max_iterations
    ):
        self.encoder_feat_size = encoder_feat_size
        self.sample_range = sample_range
        self.max_iterations = max_iterations

    def get_mask_scores(self, points, masks, all_points, emd_cost, ref_masks_pool):

        def is_in_mask(point, mask):
            # input: point: n*2, mask: h*w
            # output: n*1
            h, w = mask.shape
            point = point.astype(int)
            point = point[:, ::-1]  # y,x
            point = np.clip(point, 0, [h - 1, w - 1])
            return mask[point[:, 0], point[:, 1]]

        ori_masks = masks
        masks = cv2.resize(
            masks[0].astype(np.float32),
            (self.encoder_feat_size, self.encoder_feat_size),
            interpolation=cv2.INTER_AREA)
        if masks.max() <= 0:
            thres = masks.max() - 1e-6
        else:
            thres = 0
        masks = masks > thres

        # 1. emd
        emd_cost_pool = emd_cost[ref_masks_pool.flatten().bool(), :][:, masks.flatten()]
        emd = ot.emd2(a=[1. / emd_cost_pool.shape[0] for i in range(emd_cost_pool.shape[0])],
                      b=[1. / emd_cost_pool.shape[1] for i in range(emd_cost_pool.shape[1])],
                      M=emd_cost_pool.cpu().numpy())
        emd_score = 1 - emd

        labels = np.ones((points.shape[0],))

        # 2. purity and coverage
        assert all_points is not None
        points_in_mask = is_in_mask(all_points, ori_masks[0])
        points_in_mask = all_points[points_in_mask]
        # here we define two metrics for local matching , purity and coverage
        # purity: points_in/mask_area, the higher means the denser points in mask
        # coverage: points_in / all_points, the higher means the mask is more complete
        mask_area = max(float(masks.sum()), 1.0)
        purity = points_in_mask.shape[0] / mask_area
        coverage = points_in_mask.shape[0] / all_points.shape[0]
        purity = torch.tensor([purity]) + 1e-6
        coverage = torch.tensor([coverage]) + 1e-6
        return purity, coverage, emd_score, points, labels, ori_masks

    def combinations(self, n, k):
        if k > n:
            return []
        if k == 0:
            return [[]]
        if k == n:
            return [[i for i in range(n)]]
        res = []
        for i in range(n):
            for j in self.combinations(i, k - 1):
                res.append(j + [i])
        return res

    def sample_points(self, points):
        # return list of arrary

        sample_list = []
        label_list = []
        for i in range(min(self.sample_range[0], len(points)), min(self.sample_range[1], len(points)) + 1):
            if len(points) > 8:
                index = [random.sample(range(len(points)), i) for j in range(self.max_iterations)]
            else:
                index = self.combinations(len(points), i)

            sample = np.take(points.cpu().numpy(), index, axis=0)

            # generate label  max_iterations * i
            label = np.ones((sample.shape[0], i))
            sample_list.append(sample)
            label_list.append(label)
        return sample_list, label_list


class Matcher:
    def __init__(
        self,
        input_size=518,
        encoder_patch_size=14,
        num_centers=8,
        use_box=False,
        use_points_or_centers=True,
        sample_range=(4, 6),
        max_sample_iterations=30,
        alpha=1.,
        beta=0.,
        exp=0.,
        num_merging_mask=10,
        score_filter_cfg={},
    ):
        self.input_size = (input_size, input_size)
        self.encoder_feat_size = input_size // encoder_patch_size
        self.patch_size = encoder_patch_size
        self.num_centers = num_centers
        self.use_box = use_box
        self.use_points_or_centers = use_points_or_centers
        self.sample_range = sample_range
        self.max_sample_iterations = max_sample_iterations

        self.alpha = alpha
        self.beta = beta
        self.exp = exp

        self.num_merging_mask = num_merging_mask
        self.score_filter_cfg = score_filter_cfg

        self.rps = RobustPromptSampler(
            encoder_feat_size=self.encoder_feat_size,
            sample_range=self.sample_range,
            max_iterations=self.max_sample_iterations
        )

    def matching(self, ref_feats, tar_feat, ref_masks, device):
        # Here the batch size is set to 1 by default
        all_points, box, sim_mat, dist_mat, _ = self.patch_level_matching(
            ref_feats, tar_feat, ref_masks, device
        )
        points = self.clustering(all_points) if not self.use_points_or_centers else all_points
        return points, box, all_points, dist_mat

    def clustering(self, points):
        num_centers = min(self.num_centers, len(points))
        flag = True
        while (flag):
            centers, cluster_assignment = kmeans_pp(points, num_centers)
            id, fre = torch.unique(cluster_assignment, return_counts=True)
            if id.shape[0] == num_centers:
                flag = False
            else:
                print('Kmeans++ failed, re-run')
        centers = np.array(centers).astype(np.int64)
        return centers

    def patch_level_matching(self, ref_feats, tar_feat, ref_masks,  device):
        # forward matching
        S = ref_feats @ tar_feat.t()  # [n_shot * N, N], similarity matrix between ref and tar
        C = (1 - S) / 2  # distance between (0, 1)

        S_forward = S[ref_masks.flatten().bool()]

        indices_forward = linear_sum_assignment(S_forward.cpu(), maximize=True)
        indices_forward = [torch.as_tensor(index, dtype=torch.int64, device=device) for index in indices_forward]
        sim_scores_f = S_forward[indices_forward[0], indices_forward[1]]
        indices_mask = ref_masks.flatten().nonzero()[:, 0]

        # reverse matching
        S_reverse = S.t()[indices_forward[1]]
        indices_reverse = linear_sum_assignment(S_reverse.cpu(), maximize=True)
        indices_reverse = [torch.as_tensor(index, dtype=torch.int64, device=device) for index in indices_reverse]
        retain_ind = torch.isin(indices_reverse[1], indices_mask)
        if not (retain_ind == False).all().item():
            indices_forward = [indices_forward[0][retain_ind], indices_forward[1][retain_ind]]
            sim_scores_f = sim_scores_f[retain_ind]
        inds_matched, sim_matched = indices_forward, sim_scores_f

        reduced_points_num = len(sim_matched) // 2 if len(sim_matched) > 40 else len(sim_matched)
        sim_sorted, sim_idx_sorted = torch.sort(sim_matched, descending=True)
        sim_filter = sim_idx_sorted[:reduced_points_num]
        points_matched_inds = indices_forward[1][sim_filter]

        # points_matched_inds_set = torch.tensor(list(set(points_matched_inds.cpu().tolist())))
        points_matched_inds_set = torch.unique(points_matched_inds)

        points_matched_inds_set_w = points_matched_inds_set % self.encoder_feat_size
        points_matched_inds_set_h = points_matched_inds_set // self.encoder_feat_size
        idxs_mask_set_x = (points_matched_inds_set_w * self.patch_size + self.patch_size // 2)
        idxs_mask_set_y = (points_matched_inds_set_h * self.patch_size + self.patch_size // 2)
        points = torch.stack((idxs_mask_set_x, idxs_mask_set_y), dim=-1)

        if self.use_box:
            box = torch.stack(
                [
                    idxs_mask_set_x.min()[0],
                    idxs_mask_set_y.min()[1],
                    idxs_mask_set_x.min()[0],
                    idxs_mask_set_y.min()[1],
                ],
                dim=-1
            )
        else:
            box = None

        return points, box, S, C, reduced_points_num

    def _check_mask_shape(self, pred_masks):
        if len(pred_masks.shape) < 3:  # avoid only one mask
            pred_masks = pred_masks[None, ...]
        return pred_masks

    def filter_masks(self, tar_masks_ori, ref_masks, samples_list, points, all_points, dist_mat, device):
        tar_masks = tar_masks_ori.unsqueeze(dim=1).cpu().numpy()
        points = points.cpu().numpy()
        all_points = all_points.cpu().numpy()

        # append to original results
        purity = torch.zeros(tar_masks.shape[0])
        coverage = torch.zeros(tar_masks.shape[0])
        emd = torch.zeros(tar_masks.shape[0])

        samples = samples_list[-1]
        labels = torch.ones(tar_masks.shape[0], samples.shape[1])
        samples = torch.ones(tar_masks.shape[0], samples.shape[1], 2)

        # compute scores for each mask
        for i in range(len(tar_masks)):
            purity_, coverage_, emd_, sample_, label_, mask_ = \
                self.rps.get_mask_scores(
                    points=points,
                    masks=tar_masks[i],
                    all_points=all_points,
                    emd_cost=dist_mat,
                    ref_masks_pool=ref_masks
                )
            assert np.all(mask_ == tar_masks[i])
            purity[i] = purity_
            coverage[i] = coverage_
            emd[i] = emd_

        pred_masks = tar_masks.squeeze(1)
        metric_preds = {
            "purity": purity,
            "coverage": coverage,
            "emd": emd
        }

        scores = self.alpha * emd + self.beta * purity * coverage ** self.exp

        pred_masks = self._check_mask_shape(pred_masks)

        # filter the false-positive mask fragments by using the proposed metrics
        for metric in ["coverage", "emd", "purity"]:
            if self.score_filter_cfg[metric] > 0:
                thres = min(self.score_filter_cfg[metric], metric_preds[metric].max())
                idx = torch.where(metric_preds[metric] >= thres)[0]
                scores = scores[idx]
                samples = samples[idx]
                labels = labels[idx]
                pred_masks = self._check_mask_shape(pred_masks[idx])

                for key in metric_preds.keys():
                    metric_preds[key] = metric_preds[key][idx]

        #  score-based masks selection, masks merging
        if self.score_filter_cfg["score_filter"]:
            distances = 1 - scores
            distances, rank = torch.sort(distances, descending=False)
            distances_norm = distances - distances.min()
            distances_norm = distances_norm / (distances.max() + 1e-6)
            filer_dis = distances < self.score_filter_cfg["score"]
            filer_dis[..., 0] = True
            filer_dis_norm = distances_norm < self.score_filter_cfg["score_norm"]
            filer_dis = filer_dis * filer_dis_norm

            pred_masks = self._check_mask_shape(pred_masks)
            masks = pred_masks[rank[filer_dis][:self.num_merging_mask]]
            scores = scores[rank[filer_dis][:self.num_merging_mask]]
            masks = self._check_mask_shape(masks)
        else:
            raise NotImplementedError
            topk = min(self.num_merging_mask, scores.size(0))
            topk_idx = scores.topk(topk)[1]
            topk_samples = samples[topk_idx].cpu().numpy()
            topk_scores = scores[topk_idx].cpu().numpy()
            topk_pred_masks = pred_masks[topk_idx]
            topk_pred_masks = self._check_mask_shape(topk_pred_masks)

            if self.score_filter_cfg["topk_scores_threshold"] > 0:
                # map scores to 0-1
                topk_scores = topk_scores / (topk_scores.max())

            idx = topk_scores > self.score_filter_cfg["topk_scores_threshold"]
            topk_samples = topk_samples[idx]

            topk_pred_masks = self._check_mask_shape(topk_pred_masks)
            topk_pred_masks = topk_pred_masks[idx]
            mask_list = []
            for i in range(len(topk_samples)):
                mask = topk_pred_masks[i][None, ...]
                mask_list.append(mask)
            masks = np.sum(mask_list, axis=0) > 0
            masks = self._check_mask_shape(masks)
        return torch.tensor(masks, device=device, dtype=torch.float), torch.tensor(scores, device=device, dtype=torch.float)



class Sam2Matcher(nn.Module):
    def __init__(
        self,
        sam2_cfg_file,
        sam2_ckpt_path,
        sam2_amg_cfg,
        encoder_cfg,
        encoder_ckpt_path,
        matcher_cfg,
        memory_bank_cfg
    ):
        super(Sam2Matcher, self).__init__()

        self.with_dense_pred = matcher_cfg.pop("with_dense_pred")

        # Models
        self.sam_model = build_sam2(sam2_cfg_file, sam2_ckpt_path)
        self.sam_amg = SAM2AutomaticMaskGenerator_Matcher(
            self.sam_model, **sam2_amg_cfg
        )

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
        matcher_cfg["input_size"] = encoder_img_size
        matcher_cfg["encoder_patch_size"] = encoder_patch_size

        self.matcher = Matcher(**matcher_cfg)

        memory_bank_cfg["feat_shape"] = (self.encoder_h * self.encoder_w, self.encoder_dim)
        self._init_memory_bank(memory_bank_cfg)

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
            self.has_memory_bank = True
        else:
            self.has_memory_bank = False

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

    def forward_test(self, input_dicts):
        assert self.has_memory_bank
        assert len(input_dicts) == 1

        device = self.sam_model.device

        tar_img = input_dicts[0]["target_img"]
        tar_img_np = tar_img.permute(1, 2, 0).cpu().numpy()
        tar_img = tar_img.unsqueeze(dim=0).to(device=device)

        tar_img = self.encoder_transform(tar_img)
        tar_feat = self._forward_encoder(tar_img, normalize=True).reshape(-1, self.encoder_dim)

        self.sam_amg.predictor.set_image(tar_img_np)

        if self.with_dense_pred:
            tar_masks_dense = self.sam_amg.generate(tar_img_np, dense_pred=True)
        else:
            tar_masks_dense = None

        masks_all_classes = []
        scores_all_classes = []
        labels_all_classes = []
        for c_ind in range(self.mem_n_classes):
            ref_feats_n_shot = self.mem_feats[c_ind].reshape(-1, self.encoder_dim)
            ref_masks_n_shot = self.mem_masks[c_ind].reshape(-1)
            points, box, all_points, dist_mat = self.matcher.matching(
                ref_feats_n_shot, tar_feat, ref_masks_n_shot, device
            )
            samples_list, label_list = self.matcher.rps.sample_points(points)

            # NOTE: Masks are generated using DINOv2's input size (518, 518)
            masks_class, ious_class = self.sam_amg.generate(
                tar_img_np,
                select_point_coords=samples_list,
                select_point_labels=label_list,
                select_box=[box] if self.matcher.use_box else None,
                dense_pred=False,
                extra_mask_data=tar_masks_dense
            )

            final_masks_class, final_scores_class = self.matcher.filter_masks(
                masks_class, ref_masks_n_shot, samples_list, points, all_points, dist_mat, device
            )

            print(c_ind)

            final_labels_class = torch.zeros_like(final_scores_class).to(dtype=torch.long) + c_ind
            masks_all_classes.append(final_masks_class)
            scores_all_classes.append(final_scores_class)
            labels_all_classes.append(final_labels_class)

        self.sam_amg.predictor.reset_predictor()

        masks = torch.cat(masks_all_classes, dim=0)
        scores = torch.cat(scores_all_classes, dim=0)
        labels = torch.cat(scores_all_classes, dim=0)

        n_out = min(masks.shape[0], 100)
        inds = torch.argsort(scores, descending=True)[:n_out]

        masks = masks[inds]
        scores = scores[inds]
        labels = labels[inds]

        # resizing and converting to output format
        ori_h = input_dicts[0]["target_img_info"]["ori_height"]
        ori_w = input_dicts[0]["target_img_info"]["ori_width"]

        masks_binary = F.interpolate(
            masks.unsqueeze(dim=1),
            size=(ori_h, ori_w),
            mode="nearest"
        ).squeeze(dim=1) > 0

        bboxes = batched_mask_to_box(masks_binary)
        output_dict = dict(
            binaay_masks=masks_binary,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
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
