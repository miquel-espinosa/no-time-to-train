from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import batched_nms

from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import fill_holes_in_mask_scores, concat_points
from sam2.utils.amg import batched_mask_to_box

from dev_hongyi.models.model_utils import concat_all_gather
from dev_hongyi.utils import print_dict


class SAM2Ref(nn.Module):
    def __init__(
        self,
        sam2_cfg,
        checkpoint_path,
        memory_bank_cfg=None,
        disable_custom_iou_embed=False,
        skip_custom_iou_in_attn=True,
        semantic_ref=True,
        add_semantic_token=False,
        enable_memory_bank=False,
        use_cls_loss=False,
        testing_cfg=None,
    ):
        super(SAM2Ref, self).__init__()
        # SAM2 predictor model
        self.predictor = build_sam2_video_predictor(sam2_cfg, checkpoint_path)
        self.image_size = self.predictor.image_size
        self.semantic_ref = semantic_ref
        self.use_cls_loss = use_cls_loss
        self.add_semantic_token = add_semantic_token
        if add_semantic_token:
            assert NotImplementedError("The semantic token idea is still under construction")

        # Testing settings
        if testing_cfg is not None:
            self.testing_point_bs    = testing_cfg.get("point_bs")
            self.testing_nms_iou_thr = testing_cfg.get("nms_iou_thr")
            self.testing_out_num     = testing_cfg.get("max_keep_num")

        for param in self.predictor.parameters():
            param.requires_grad = False  # Freeze all parameters

        # Memory Encoder
        self.mem_feat_ref_pe = torch.nn.Embedding(1, self.predictor.mem_dim)
        if not self.semantic_ref:
            self.mem_ptr_ref_pe = torch.nn.Embedding(1, self.predictor.mem_dim)

        self.iou_prediction_head = MLP(
            input_dim=self.predictor.sam_mask_decoder.transformer_dim,
            hidden_dim=self.predictor.sam_mask_decoder.iou_head_hidden_dim,
            output_dim=self.predictor.sam_mask_decoder.num_mask_tokens,
            num_layers=self.predictor.sam_mask_decoder.iou_head_depth,
            sigmoid_output=self.predictor.sam_mask_decoder.iou_prediction_use_sigmoid
        )
        self.disable_custom_iou_embed = disable_custom_iou_embed
        if not disable_custom_iou_embed:
            self.iou_embed = nn.Embedding(1, self.predictor.sam_mask_decoder.transformer_dim)


        # Semantic token
        if self.add_semantic_token:
            self.semantic_token_proj = MLP(
                input_dim=256,
                hidden_dim=1024,
                output_dim=self.predictor.sam_mask_decoder.transformer_dim,
                num_layers=2
            )

        self.n_skip_tokens_in_attn = 0
        if not add_semantic_token:
            self.n_skip_tokens_in_attn += 1
        if not disable_custom_iou_embed:
            self.n_skip_tokens_in_attn += 1
        if not skip_custom_iou_in_attn:
            self.n_skip_tokens_in_attn = 0

        # Memory bank
        if enable_memory_bank:
            self.mem_n_category    = memory_bank_cfg.get("category_num")
            self.mem_length        = memory_bank_cfg.get("length")
            self.mem_feat_size     = memory_bank_cfg.get("feat_size")
            self.mem_dim           = memory_bank_cfg.get("dimension")
            if not self.semantic_ref:
                self.mem_obj_feat_size = memory_bank_cfg.get("obj_ptr_size_size")
            self.register_buffer(
                "memory_fill_buffer",
                torch.zeros((self.mem_n_category,), dtype=torch.long)
            )
            self.register_buffer(
                "memory_bank",
                torch.zeros((self.mem_n_category, self.mem_length, self.mem_feat_size, self.mem_dim))
            )
            self.register_buffer(
                "memory_pe",
                torch.zeros((self.mem_feat_size, self.mem_dim,))
            )
            if not self.semantic_ref:
                self.register_buffer(
                    "obj_ptr_bank",
                    torch.zeros((self.mem_n_category, self.mem_length, self.mem_obj_feat_size, self.mem_dim))
                )
                self.register_buffer(
                    "obj_ptr_pe",
                    torch.zeros((self.mem_obj_feat_size, self.mem_dim,))
                )
            if self.add_semantic_token:
                self.register_buffer(
                    "memory_semantic_tokens",
                    torch.zeros((self.mem_n_category, self.predictor.sam_mask_decoder.transformer_dim))
                )
            self.has_memory_bank = True
        else:
            self.has_memory_bank = False

    def _encode_image(self, img):
        with torch.inference_mode():
            backbone_out = self.predictor.forward_image(img)
            # Backbone output has keys: 'backbone_fpn', 'vision_pos_enc', 'vision_features'

            _, img_vision_features, img_vision_pos_embeds, img_feature_sizes = (
                self.predictor._prepare_backbone_features(backbone_out)
            )
            return img_vision_features, img_vision_pos_embeds, img_feature_sizes

    def _fill_holes(self, pred_masks):
        """ Fill holes in the predicted masks """
        if self.predictor.fill_hole_area > 0:
            pred_masks = fill_holes_in_mask_scores(pred_masks, self.predictor.fill_hole_area)
        return pred_masks.to(self.predictor.device, non_blocking=True)

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

    def _get_ref_nums_by_cat(self, input_dicts):
        ret = []
        total_num = 0
        for d in input_dicts:
            ret.append(OrderedDict())
            for cat_ind in d["refs_by_cat"].keys():
                ret[-1][cat_ind] = len(d["refs_by_cat"][cat_ind]['img_info'])
                total_num += len(d["refs_by_cat"][cat_ind]['img_info'])
        return ret, total_num

    def _separate_ref_tensors(self, ref_xs, ref_nums_by_cat, total_ref_num):
        separated_ref_xs = []
        for x in ref_xs:
            assert x.shape[0] == total_ref_num
            x_batched = []
            curr_pos = 0
            for rn in ref_nums_by_cat:
                x_batched.append(OrderedDict())
                for cat_ind in rn.keys():
                    x_batched[-1][cat_ind] = x[curr_pos:curr_pos+rn[cat_ind]]
                    curr_pos += rn[cat_ind]
            separated_ref_xs.append(x_batched)
        return separated_ref_xs

    def _stack_all_images(self, input_dicts, ref_nums_by_cat):
        images = []
        # add target images
        for d in input_dicts:
            images.append(d["target_img"].unsqueeze(dim=0).to(device=self.predictor.device))
        # add reference images
        for d, rn in zip(input_dicts, ref_nums_by_cat):
            for cat_ind in rn.keys():
                images.append(d["refs_by_cat"][cat_ind]["imgs"].to(device=self.predictor.device))
        images_all = torch.cat(images, dim=0)
        return images_all

    def _stack_ref_masks(self, input_dicts, ref_nums_by_cat):
        masks = []
        for d, rn in zip(input_dicts, ref_nums_by_cat):
            for cat_ind in rn.keys():
                masks.append(d["refs_by_cat"][cat_ind]["masks"].to(device=self.predictor.device))
        ref_masks_all = torch.cat(masks, dim=0)
        return ref_masks_all.unsqueeze(dim=1)

    def _stack_points(self, input_dicts, ref_nums_by_cat):
        points = []
        n_points_per_cat = 0
        for d, rn in zip(input_dicts, ref_nums_by_cat):
            for cat_ind in rn.keys():
                points.append(d["tar_anns_by_cat"][cat_ind]["query_points"].to(device=self.predictor.device).reshape(1, -1, 2))
                if n_points_per_cat == 0:
                    n_points_per_cat = points[0].shape[1]
        all_points = torch.cat(points, dim=0)
        return all_points, n_points_per_cat

    def _forward_references(self, ref_img_feats, ref_img_pes, feat_sizes, ref_binary_masks):
        '''
        A simplified version of SAM2Base.track_step
        '''

        if len(ref_img_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(ref_img_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        pix_feat = ref_img_feats[-1].permute(1, 2, 0)
        pix_feat = pix_feat.view(-1, self.predictor.hidden_dim, *feat_sizes[-1])

        if not self.semantic_ref:
            _, _, _, low_res_masks, high_res_masks, obj_ptrs, _ = self.predictor._use_mask_as_output(
                pix_feat, high_res_features, ref_binary_masks
            )
        else:
            out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
            mask_inputs_float = ref_binary_masks.float()
            high_res_masks = mask_inputs_float * out_scale + out_bias
            obj_ptrs = torch.zeros(
                ref_binary_masks.size(0), self.predictor.hidden_dim, device=self.predictor.device
            )

        maskmem_features, maskmem_pos_enc = self.predictor._encode_new_memory(
            current_vision_feats=ref_img_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=False,
            force_binarize=True
        )
        maskmem_features = torch.cat([x.unsqueeze(dim=0) for x in maskmem_features], dim=0)
        return maskmem_features, maskmem_pos_enc[0], obj_ptrs

    def _forward_memory_forLoop(
            self,
            ref_maskmem_feats,
            ref_maskmem_pes,
            ref_obj_ptrs,
            ref_nums_by_cat,
            total_ref_num,
            tar_feat,
            tar_pe,
            feat_size
    ):
        # A simplified version of SAM2Base._prepare_memory_conditioned_features

        batch_size = tar_feat.shape[1]  # tar_feat [hw, B, C]
        mem_c = self.predictor.mem_dim

        ref_maskmem_feats_sep, ref_maskmem_pes_sep, ref_obj_ptrs_sep = self._separate_ref_tensors(
            [ref_maskmem_feats, ref_maskmem_pes, ref_obj_ptrs], ref_nums_by_cat, total_ref_num
        )

        pix_feats_with_mem = []
        for i in range(batch_size):
            feats_i = []
            for cat_ind in ref_nums_by_cat[i].keys():

                feat_memory = ref_maskmem_feats_sep[i][cat_ind].flatten(2).permute(2, 0, 1).reshape(-1, 1, mem_c)
                feat_memory_pe = ref_maskmem_feats_sep[i][cat_ind].flatten(2).permute(2, 0, 1).reshape(-1, 1, mem_c)
                feat_memory_pe = feat_memory_pe + self.mem_feat_ref_pe.weight.reshape(1, 1, -1)

                if not self.semantic_ref:
                    obj_ptr_memory = ref_obj_ptrs_sep[i][cat_ind].reshape(-1, 1, mem_c)
                    obj_ptr_memory_pe = torch.zeros_like(obj_ptr_memory) + self.mem_ptr_ref_pe.weight.reshape(1, 1, -1)
                    num_obj_ptr_tokens = obj_ptr_memory.shape[0]
                    memory = torch.cat((feat_memory, obj_ptr_memory), dim=0)
                    memory_pe = torch.cat((feat_memory_pe, obj_ptr_memory_pe), dim=0)
                else:
                    num_obj_ptr_tokens = 0
                    memory = feat_memory
                    memory_pe = feat_memory_pe

                pix_feat_with_mem = self.predictor.memory_attention(
                    curr=tar_feat[:, i:i+1, :],
                    curr_pos=tar_pe[:, i:i+1, :],
                    memory=memory,
                    memory_pos=memory_pe,
                    num_obj_ptr_tokens=num_obj_ptr_tokens,
                ).permute(1, 2, 0).view(1, -1, feat_size[0], feat_size[1])
                feats_i.append(pix_feat_with_mem)
            pix_feats_with_mem.append(torch.cat(feats_i, dim=0))
        return pix_feats_with_mem

    def _forward_memory_testing(self, tar_feat, tar_pe, feat_size):
        mem_c = self.mem_dim
        memory = (
            self.memory_bank
            .reshape(self.mem_n_category, -1, mem_c)
            .permute(1, 0, 2)
        )  # [length *  mem_feat_size, n_cat,  mem_dim]
        memory_pe = (
            self.memory_pe
            .reshape(1, 1, self.mem_feat_size, mem_c)
            .expand(self.mem_n_category, self.mem_length, -1, -1)
            .reshape(self.mem_n_category, -1, mem_c)
            .permute(1, 0, 2)
        )  # [length *  mem_feat_size, n_cat,  mem_dim]
        if not self.semantic_ref:
            obj_ptrs = (
                self.obj_ptr_bank
                .reshape(self.mem_n_category, -1, mem_c)
                .permute(1, 0, 2)
            )  # [length * mem_obj_ptr_size, n_cat, mem_dim]
            obj_pe = (
                self.obj_ptr_pe
                .reshape(1, 1, self.mem_obj_feat_size, mem_c)
                .expand(self.mem_n_category, self.mem_length, -1, -1)
                .reshape(self.mem_n_category, -1, mem_c)
                .permute(1, 0, 2)
            )  # [length *  mem_obj_ptr_size, n_cat,  mem_dim]

            num_obj_ptr_tokens = obj_ptrs.shape[0]
            memory_all = torch.cat((memory, obj_ptrs), dim=0)
            memory_pe_all = torch.cat((memory_pe, obj_pe), dim=0)
        else:
            num_obj_ptr_tokens = 0
            memory_all = memory
            memory_pe_all = memory_pe

        tar_feat = tar_feat.expand(-1, self.mem_n_category, -1)
        tar_pe = tar_pe.expand(-1, self.mem_n_category, -1)

        pix_feat_with_mem = self.predictor.memory_attention(
            curr=tar_feat,
            curr_pos=tar_pe,
            memory=memory_all,
            memory_pos=memory_pe_all,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        ).permute(1, 2, 0).reshape(self.mem_n_category, -1, feat_size[0], feat_size[1])
        return pix_feat_with_mem

    def _forward_decoder_testing(
            self,
            backbone_features,
            backbone_hr_features,
            point_inputs,
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
        custom_iou_token = self.iou_embed.weight.expand(B, -1, -1)
        sparse_embeddings = torch.cat([sparse_embeddings, custom_iou_token], dim=1)

        (
            low_res_multimasks,
            ious,
            _,
            _,
            custom_iou_token_out
        ) = self.predictor.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=backbone_hr_features,
            return_iou_token_out=True,
            disable_custom_iou_embed=self.disable_custom_iou_embed,
            disable_mlp_obj_scores=True,
            output_all_masks=True,
            skip_last_n_keys=self.n_skip_tokens_in_attn
        )

        custom_iou_pred = self.iou_prediction_head(custom_iou_token_out).reshape(B, -1)
        # --------
        custom_iou_pred = ious * custom_iou_pred
        # --------
        n_pred = custom_iou_pred.shape[-1]
        assert n_pred == low_res_multimasks.shape[1]

        # We skip the SAM2's multimask_output but use the custom IoU to determine the output mask
        # TODO: add advanced mask postprocessing tricks in the sam2 auto mask generator
        best_iou_inds = torch.argmax(custom_iou_pred, dim=-1)
        batch_inds = torch.arange(B, device=device)
        low_res_masks = low_res_multimasks[batch_inds, best_iou_inds]
        scores = custom_iou_pred[batch_inds, best_iou_inds]

        return low_res_masks, scores

    def forward_tar_ref(self, input_dicts):
        '''
        each one in input_dicts:
            target_img: [3, 1024, 1024]
            target_img_info: a dict containing ori_height, ori_width, file_name, id (COCO format)
            tar_anns_by_cat: (empty when eval)
                each cat_ind:
                    masks: torch.Size([n_ins, 1024, 1024])
                    bboxes: <class 'torch.Tensor'> torch.Size([n_ins, 4])
                    query_points: <class 'torch.Tensor'> torch.Size([n_query_points, 2])
            refs_by_cat:
                each cat_ind:
                    img_info: list containing meta info of each reference image
                    imgs:  [n_ref, 3, 1024, 1024]
                    masks: [n_ref, 1024, 1024]
                    bboxes: [n_ref, 4]
        '''
        assert type(input_dicts) is list

        batch_size = len(input_dicts)

        ref_nums_by_cat, total_ref_num = self._get_ref_nums_by_cat(input_dicts)

        # encode image features
        images_all = self._stack_all_images(input_dicts, ref_nums_by_cat)
        img_feats, img_pes, img_feat_sizes = self._encode_image(images_all)
        '''
        img_feats: list of [stride4, stride8, stride16] features, each [hw, B, C_stride]
        img_pes: list of [stride4, stride8, stride16] PE, each [hw, B, 256]
        img_feat_sizes: [(256, 256), (128, 128), (64, 64)]
        '''
        tar_img_feats = [x[:, :batch_size, :] for x in img_feats]
        tar_img_pes = [x[:, :batch_size, :] for x in img_pes]
        ref_img_feats = [x[:, batch_size:, :] for x in img_feats]
        ref_img_pes = [x[:, batch_size:, :] for x in img_pes]

        # forward references
        ref_binary_masks = self._stack_ref_masks(input_dicts, ref_nums_by_cat)  # [n_ref_all, 1, 1024, 1024]
        ref_binary_masks = ref_binary_masks.to(dtype=ref_img_feats[0].dtype)

        ref_maskmem_feats, ref_maskmem_pes, ref_obj_ptrs = self._forward_references(
            ref_img_feats, ref_img_pes, img_feat_sizes, ref_binary_masks
        )
        # ref_maskmem_feats = ref_maskmem_feats.to(torch.bfloat16).to(self.predictor.device, non_blocking=True)
        if ref_maskmem_pes is not None:
            assert ref_maskmem_pes.shape[0] == ref_maskmem_feats.shape[0]
            expanded_maskmem_pes = ref_maskmem_pes
        else:
            expanded_maskmem_pes = None

        tar_feats_with_mem = self._forward_memory_forLoop(
            ref_maskmem_feats,
            expanded_maskmem_pes,
            ref_obj_ptrs,
            ref_nums_by_cat,
            total_ref_num,
            tar_img_feats[-1],
            tar_img_pes[-1],
            img_feat_sizes[-1]
        )
        tar_feats_with_mem_batched = torch.cat(tar_feats_with_mem, dim=0)   # [n_cat * B, C, h, w]

        # # ---------------------------------------------------------
        # # Not conditioning on memory, for debugging
        # _tar_feats = tar_img_feats[-1].permute(1, 2, 0).reshape(batch_size, -1, img_feat_sizes[-1][0], img_feat_sizes[-1][1])
        # _tar_feats_expand = []
        # for i in range(batch_size):
        #     fi = _tar_feats[i:i+1]
        #     fi_expanded = fi.expand(len(input_dicts[i]["tar_anns_by_cat"].keys()), -1, -1, -1)
        #     _tar_feats_expand.append(fi_expanded)
        # tar_feats_with_mem_batched = torch.cat(_tar_feats_expand, dim=0)
        # # ---------------------------------------------------------

        tar_high_res_feats = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(tar_img_feats[:-1], img_feat_sizes[:-1])
        ]
        expanded_hr_feats_batched = []
        for hr_feat in tar_high_res_feats:
            expanded_hr_feat = []
            for i in range(batch_size):
                expanded_hr_feat.append(
                    hr_feat[i:i+1].expand(len(input_dicts[i]["tar_anns_by_cat"].keys()), -1, -1, -1)
                )
            expanded_hr_feats_batched.append(torch.cat(expanded_hr_feat, dim=0))

        stacked_points, n_points_per_cat = self._stack_points(input_dicts, ref_nums_by_cat)
        stacked_points = stacked_points.reshape(-1, 1, 2)
        point_labels = torch.ones_like(stacked_points[:, :, 0:1]).to(dtype=torch.int32)
        tar_point_inputs = dict(
            point_coords=stacked_points,
            point_labels=point_labels.reshape(-1, 1)
        )

        bs_decoder = tar_feats_with_mem_batched.shape[0] * n_points_per_cat
        tar_feats_with_mem_batched = (
            tar_feats_with_mem_batched
            .unsqueeze(dim=1)
            .expand(-1, n_points_per_cat, -1, -1, -1)
            .reshape(bs_decoder, *tar_feats_with_mem_batched.shape[-3:])
        )
        expanded_hr_feats_batched = [
            (
                x
                .unsqueeze(dim=1)
                .expand(-1, n_points_per_cat, -1, -1, -1)
                .reshape(bs_decoder, *x.shape[-3:])
            )
            for x in expanded_hr_feats_batched
        ]

        if self.disable_custom_iou_embed:
            custom_iou_token = None
        else:
            custom_iou_token = self.iou_embed.weight.expand(bs_decoder, -1, -1)

        (
            tar_low_res_multimasks,
            tar_high_res_multimasks,
            my_iou_token_out,  # my iou token out, used for passing into my MLP
            tar_ious,  # original predicted ious
            tar_low_res_masks,
            tar_high_res_masks,
            tar_obj_ptr,
            tar_object_score_logits  # output of the occlusion MLP
        ) = self.predictor._forward_sam_heads(
            backbone_features=tar_feats_with_mem_batched,
            point_inputs=tar_point_inputs,
            mask_inputs=None,
            high_res_features=expanded_hr_feats_batched,
            multimask_output=True,
            return_iou_token_out=True,
            merge_sparse_with_my_token=custom_iou_token,
            disable_custom_iou_embed=self.disable_custom_iou_embed,
            disable_mlp_obj_scores=True,
            output_all_masks=True,
            skip_last_n_keys=self.n_skip_tokens_in_attn
        )
        # tar_pred_masks = self._fill_holes(tar_low_res_masks)
        custom_iou_pred = self.iou_prediction_head(my_iou_token_out)
        return tar_low_res_masks, custom_iou_pred, ref_nums_by_cat, n_points_per_cat

    def forward_train(self, input_dicts):
        tar_pred_masks, custom_iou_pred, ref_nums_by_cat, n_points_per_cat = self.forward_tar_ref(input_dicts)

        tar_pred_masks = tar_pred_masks.reshape(
            -1, n_points_per_cat, *tar_pred_masks.shape[-3:]
        )  # [B * n_cat, n_points, 4, 256, 256]
        custom_iou_pred = custom_iou_pred.reshape(
            -1, n_points_per_cat, custom_iou_pred.shape[-1]
        )  # [B * n_cat, n_points, 4]

        n_output, pred_mask_h, pred_mask_w = tar_pred_masks.shape[-3:]

        gt_masks_bool = []
        for d, rn in zip(input_dicts, ref_nums_by_cat):
            for cat_ind in rn.keys():
                gt_masks_cat = d['tar_anns_by_cat'][cat_ind]['masks'].to(device=self.predictor.device)
                gt_masks_cat = gt_masks_cat.unsqueeze(dim=1).to(dtype=torch.float)
                gt_masks_cat = F.interpolate(
                    gt_masks_cat,
                    size=(pred_mask_h, pred_mask_w),
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )  # [n_ins, 1, 256, 256]
                gt_masks_bool.append(gt_masks_cat > 0)
        tar_pred_masks_bool = tar_pred_masks > 0

        matched_iou = self._compute_matched_iou_matrix(gt_masks_bool, tar_pred_masks_bool)
        if not self.use_cls_loss:
            iou_loss = torch.abs(matched_iou.reshape(-1) - custom_iou_pred.reshape(-1)).mean()
            losses = {'iou_loss': iou_loss}
        else:
            cls_loss = F.binary_cross_entropy(
                input=custom_iou_pred.reshape(-1),
                target=(matched_iou.reshape(-1) > 0.5).to(dtype=torch.int32),
                reduction='mean'
            )
            losses = {'cls_loss': cls_loss}
        metrics = {"mean_seg_iou": matched_iou.mean()}
        return losses, metrics

    def forward_fill_memory(self, input_dicts):
        '''
            Filling memory banks with references in different categories
        '''
        with torch.inference_mode():
            assert len(input_dicts) == 1
            ref_cat_ind = list(input_dicts[0]["refs_by_cat"].keys())[0]

            ref_img_feats, ref_img_pes, img_feat_sizes = self._encode_image(
                input_dicts[0]["refs_by_cat"][ref_cat_ind]["imgs"].to(device=self.predictor.device)
            )

            ref_binary_masks = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_img_feats[0].dtype)
            ref_binary_masks = input_dicts[0]["refs_by_cat"][ref_cat_ind]["masks"].to(dtype=ref_img_feats[0].dtype)
            ref_maskmem_feats, ref_maskmem_pes, ref_obj_ptrs = self._forward_references(
                ref_img_feats, ref_img_pes, img_feat_sizes, ref_binary_masks.unsqueeze(dim=0)
            )

            mem_c = self.mem_dim
            feat_memory = ref_maskmem_feats.flatten(2).permute(2, 0, 1).reshape(-1, 1, mem_c)
            feat_memory_pe = ref_maskmem_pes.flatten(2).permute(2, 0, 1).reshape(-1, 1, mem_c)
            feat_memory_pe = feat_memory_pe + self.mem_feat_ref_pe.weight.reshape(1, 1, -1)

            memory = feat_memory.reshape(1, self.mem_feat_size, mem_c)
            memory_pe = feat_memory_pe.reshape(self.mem_feat_size, mem_c)
            memory_all = concat_all_gather(memory.contiguous())

            if not self.semantic_ref:
                obj_ptr_memory = ref_obj_ptrs.reshape(-1, 1, mem_c)
                obj_ptr_memory_pe = torch.zeros_like(obj_ptr_memory) + self.mem_ptr_ref_pe.weight.reshape(1, 1, -1)
                obj_ptr_memory = obj_ptr_memory.reshape(1, self.mem_obj_feat_size, mem_c)
                obj_ptr_memory_pe = obj_ptr_memory_pe.reshape(self.mem_obj_feat_size, mem_c)
                obj_ptr_memory_all = concat_all_gather(obj_ptr_memory.contiguous())

            cat_ind_tensor = torch.tensor([ref_cat_ind], dtype=torch.long, device=self.predictor.device).reshape(1, 1)
            cat_ind_all = concat_all_gather(cat_ind_tensor).reshape(-1).to(dtype=torch.long).detach()

            for i in range(cat_ind_all.shape[0]):
                fill_ind = self.memory_fill_buffer[cat_ind_all[i]]
                self.memory_bank[cat_ind_all[i], fill_ind] = (
                        self.memory_bank[cat_ind_all[i], fill_ind] * 0.0 + memory_all[i]
                )
                if not self.semantic_ref:
                    self.obj_ptr_bank[cat_ind_all[i], fill_ind] = (
                            self.obj_ptr_bank[cat_ind_all[i], fill_ind] * 0.0 + obj_ptr_memory_all[i]
                    )
                self.memory_fill_buffer[cat_ind_all[i]] += 1
            self.memory_pe[:, :] = self.memory_pe[:, :] * 0.0 + memory_pe
            if not self.semantic_ref:
                self.obj_ptr_pe[:, :] = self.obj_ptr_pe[:, :] * 0.0 + obj_ptr_memory_pe
            return {}

    def forward_test(self, input_dicts, return_iou_grid_scores=False):
        with torch.inference_mode():
            assert self.has_memory_bank
            assert not self.disable_custom_iou_embed
            assert len(input_dicts) == 1

            tar_img = input_dicts[0]["target_img"].unsqueeze(dim=0).to(device=self.predictor.device)
            tar_img_feats, tar_img_pes, img_feat_sizes = self._encode_image(tar_img)

            tar_feats_with_mem = self._forward_memory_testing(
                tar_img_feats[-1], tar_img_pes[-1], img_feat_sizes[-1]
            )  # [n_cat, c, h, w]

            # ------------------------------------------------------------------
            # Skip memory attention
            # tar_feats_with_mem = (
            #     tar_img_feats[-1]
            #     .permute(1, 2, 0)
            #     .reshape(1, -1, *img_feat_sizes[-1])
            #     .expand(self.mem_n_category, -1, -1, -1)
            # )
            # ------------------------------------------------------------------
            expanded_hr_feats = [
                x.permute(1, 2, 0).reshape(1, -1, *s).expand(self.mem_n_category, -1, -1, -1)
                for x, s in zip(tar_img_feats[:-1], img_feat_sizes[:-1])
            ]

            points = input_dicts[0]["query_points"].to(device=self.predictor.device).reshape(1, -1, 2)
            point_labels = torch.ones_like(points[:, :, 0:1]).to(dtype=torch.int32)
            n_points = points.shape[1]

            assert n_points % self.testing_point_bs == 0

            bs_decoder = self.mem_n_category * self.testing_point_bs
            tar_feats_with_mem_batched = (
                tar_feats_with_mem
                .unsqueeze(dim=1)
                .expand(-1, self.testing_point_bs, -1, -1, -1)
                .reshape(bs_decoder, *tar_feats_with_mem.shape[-3:])
            )
            tar_hr_feats_batched = [
                (
                    x
                    .unsqueeze(dim=1)
                    .expand(-1, self.testing_point_bs, -1, -1, -1)
                    .reshape(bs_decoder, *x.shape[-3:])
                )
                for x in expanded_hr_feats
            ]

            mask_scores = []
            lr_masks = []
            for i in range(0, n_points // self.testing_point_bs):
                i_start = i * self.testing_point_bs
                i_end = i_start + self.testing_point_bs
                points_i = points[:, i_start:i_end, :].expand(self.mem_n_category, -1, -1)
                p_labels_i = point_labels[:, i_start:i_end, :].expand(self.mem_n_category, -1, -1)
                point_inputs_i = dict(
                    point_coords=points_i.reshape(bs_decoder, 1, 2),
                    point_labels=p_labels_i.reshape(bs_decoder, 1)
                )

                lr_masks_i, scores_i = self._forward_decoder_testing(
                    tar_feats_with_mem_batched, tar_hr_feats_batched, point_inputs_i
                )
                mask_scores.append(scores_i.reshape(self.mem_n_category, self.testing_point_bs))
                lr_masks.append(
                    lr_masks_i.reshape(self.mem_n_category, self.testing_point_bs, *lr_masks_i.shape[-2:])
                )

            if return_iou_grid_scores:
                scores_for_iou_plotting = (
                    torch.stack(mask_scores, dim=0)
                    .permute(1, 0, 2)
                    .reshape(self.mem_n_category, -1, 1)
                )
                points_for_iou_plotting = points.expand(self.mem_n_category, -1, -1)
            else:
                scores_for_iou_plotting = None
                points_for_iou_plotting = None

            scores_all = torch.cat(mask_scores, dim=1).reshape(self.mem_n_category * n_points)
            lr_masks_all = torch.cat(lr_masks, dim=1)
            lr_masks_all = lr_masks_all.reshape(self.mem_n_category * n_points, *lr_masks_all.shape[-2:])

            # ------------------------------------------------------------------
            # Replace scores with oracle IoUs
            # lr_masks_all = lr_masks_all.reshape(self.mem_n_category, n_points, *lr_masks_all.shape[-2:])
            # tar_anns_by_cat = input_dicts[0].pop("tar_anns_by_cat")
            # scores_oracle = torch.zeros_like(scores_all).reshape(self.mem_n_category, n_points)
            # for cat_ind in tar_anns_by_cat.keys():
            #     lr_masks_cat = lr_masks_all[cat_ind].reshape(1, n_points, 1, *lr_masks_all.shape[-2:])
            #     gt_masks_cat = tar_anns_by_cat[cat_ind]["masks"].to(dtype=torch.float, device=self.predictor.device)
            #     gt_masks_cat = F.interpolate(
            #         gt_masks_cat.unsqueeze(dim=1),
            #         size=(lr_masks_cat.shape[-2], lr_masks_cat.shape[-1]),
            #         align_corners=False,
            #         mode="bilinear",
            #         antialias=True,
            #     ).squeeze(dim=1).bool()
            #     matched_iou = self._compute_matched_iou_matrix([gt_masks_cat], lr_masks_cat > 0).reshape(n_points)
            #     scores_oracle[cat_ind] += matched_iou
            # lr_masks_all = lr_masks_all.reshape(self.mem_n_category * n_points, *lr_masks_all.shape[-2:])
            # scores_all = scores_oracle.reshape(self.mem_n_category * n_points)
            # ------------------------------------------------------------------

            # use low-resolution masks in NMS to save memory
            bboxes_all_lr = batched_mask_to_box(lr_masks_all > 0)  # [n_cat * n_points, 4]

            labels_all = (
                torch.arange(self.mem_n_category, dtype=torch.int32)
                .to(device=self.predictor.device)
                .unsqueeze(dim=1)
                .expand(-1, n_points)
                .reshape(-1)
            )
            keep_by_nms = batched_nms(
                bboxes_all_lr.float(),
                scores_all,
                labels_all,
                iou_threshold=self.testing_nms_iou_thr
            )
            lr_masks_keep = lr_masks_all[keep_by_nms][:self.testing_out_num]
            bboxes_keep = bboxes_all_lr[keep_by_nms][:self.testing_out_num]
            scores_keep = scores_all[keep_by_nms][:self.testing_out_num]
            labels_keep = labels_all[keep_by_nms][:self.testing_out_num]

            ori_h = input_dicts[0]["target_img_info"]["ori_height"]
            ori_w = input_dicts[0]["target_img_info"]["ori_width"]

            # Resize bboxes
            bboxes_resized = torch.zeros_like(bboxes_keep)
            bboxes_resized[..., 0] = bboxes_keep[..., 0] * float(ori_w) / lr_masks_keep.shape[-1]
            bboxes_resized[..., 1] = bboxes_keep[..., 1] * float(ori_h) / lr_masks_keep.shape[-2]
            bboxes_resized[..., 2] = bboxes_keep[..., 2] * float(ori_w) / lr_masks_keep.shape[-1]
            bboxes_resized[..., 3] = bboxes_keep[..., 3] * float(ori_h) / lr_masks_keep.shape[-2]

            # Resize masks
            masks_ori_size = F.interpolate(
                lr_masks_keep.unsqueeze(dim=1),
                size=(ori_h, ori_w),
                mode="bilinear",
                align_corners=False,
                antialias=True
            ).squeeze(dim=1)

            output_dict = dict(
                binaay_masks=masks_ori_size > 0,
                bboxes=bboxes_resized,
                scores=scores_keep,
                labels=labels_keep,
                image_info=input_dicts[0]["target_img_info"],
                scores_for_iou_plotting=scores_for_iou_plotting,
                points_for_iou_plotting=points_for_iou_plotting
            )
            return [output_dict]

    def forward(self, input_dicts, return_iou_grid_scores=False):
        data_mode = input_dicts[0].pop("data_mode", None)
        assert data_mode is not None
        if self.training:
            assert data_mode == "train"
            return self.forward_train(input_dicts)
        else:
            if data_mode == "fill_memory":
                return self.forward_fill_memory(input_dicts)
            elif data_mode == "test":
                return self.forward_test(input_dicts, return_iou_grid_scores)
            else:
                raise NotImplementedError(f"Unrecognized data mode during inference: {data_mode}")


