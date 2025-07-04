import torch
import torch.nn as nn
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import fill_holes_in_mask_scores


class SAM2QueryTarget(nn.Module):
    def __init__(self, model_cfg, checkpoint_path, disable_custom_iou_embed=False):
        super(SAM2QueryTarget, self).__init__()
        # SAM2 predictor model
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path)
        self.image_size = self.predictor.image_size
        
        for param in self.predictor.parameters():
            param.requires_grad = False # Freeze all parameters
        
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
        
    def _encode_image(self, img):
        """ Encode the image using the SAM2 hiera backbone """
        img_backbone_out = self.predictor.forward_image(img.to(self.predictor.device))
        # Backbone output has keys: 'backbone_fpn', 'vision_pos_enc', 'vision_features'
        _, img_vision_features, img_vision_pos_embeds, img_feature_sizes = self.predictor._prepare_backbone_features(img_backbone_out)
        return img_vision_features, img_vision_pos_embeds, img_feature_sizes
    
    def _prepare_query_mask(self, query_binary_mask):
        """ Resize the query mask to the model's image size """
        mask_H, mask_W = query_binary_mask.shape[-2:] # H, W
        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                query_binary_mask,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = query_binary_mask
        return mask_inputs
    
    def _forward_query(self, query_vision_features, query_vision_pos_embeds, query_feature_sizes, query_binary_mask, output_dict):
        """ Forward the query image through SAM2 """
        return self.predictor.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=query_vision_features,
            current_vision_pos_embeds=query_vision_pos_embeds,
            feat_sizes=query_feature_sizes,
            point_inputs=None,
            mask_inputs=query_binary_mask,
            output_dict=output_dict,
            num_frames=1,
            track_in_reverse=False,
            run_mem_encoder=False, # We will encode the mask later
            prev_sam_mask_logits=None,
        )
        
    def _fill_holes(self, pred_masks):
        """ Fill holes in the predicted masks """
        if self.predictor.fill_hole_area > 0:
            pred_masks = fill_holes_in_mask_scores( pred_masks, self.predictor.fill_hole_area )
        return pred_masks.to(self.predictor.device, non_blocking=True)
    
    def forward(self, query_img, query_binary_mask, target_img, target_point_inputs):

        # query_img: torch.Size([1, 3, 1024, 1024])
        # query_binary_mask: torch.Size([1, 1, 360, 640])
        # target_img: torch.Size([1, 3, 1024, 1024])
        # target_point_inputs["target_point_inputs"]: torch.Size([32, 1, 2])

        batch_size = query_img.shape[0]
        num_points = target_point_inputs['point_coords'].shape[0]
        
        # ENCODE QUERY and TARGET IMAGE with image encoder
        query_vision_features, query_vision_pos_embeds, query_feature_sizes = self._encode_image(query_img)


        target_vision_features, target_vision_pos_embeds, target_feature_sizes = self._encode_image(target_img)
        
        
        # PREPARE OUTPUT DICT for storing needed values
        output_dict = {}
        output_dict['cond_frame_outputs'] = {}
        output_dict['non_cond_frame_outputs'] = {}
        
        # Forward query image through SAM2: reshape + decoder
        query_binary_mask = self._prepare_query_mask(query_binary_mask)
        # torch.Size([1, 1, 1024, 1024]), mask in (0, 1)

        current_out = self._forward_query(query_vision_features, query_vision_pos_embeds, query_feature_sizes,
                                          query_binary_mask, output_dict)

        '''
        current_out: dict
            point_inputs: <class 'NoneType'> UNDEFINED_FORMAT
            mask_inputs: <class 'torch.Tensor'> torch.Size([1, 1, 1024, 1024])
            pred_masks: <class 'torch.Tensor'> torch.Size([1, 1, 256, 256])
            pred_masks_high_res: <class 'torch.Tensor'> torch.Size([1, 1, 1024, 1024])
            obj_ptr: <class 'torch.Tensor'> torch.Size([1, 256])
            maskmem_features: <class 'NoneType'> UNDEFINED_FORMAT
            maskmem_pos_enc: <class 'NoneType'> UNDEFINED_FORMAT
        '''
        
        pred_masks = self._fill_holes(current_out["pred_masks"])
        output_dict['cond_frame_outputs'][0] = current_out
        output_dict['cond_frame_outputs'][0]['pred_masks'] = pred_masks
        '''
        output_dict:
            cond_frame_outputs:
                0:
                    point_inputs: <class 'NoneType'> UNDEFINED_FORMAT
                    mask_inputs: <class 'torch.Tensor'> torch.Size([1, 1, 1024, 1024])
                    pred_masks: <class 'torch.Tensor'> torch.Size([1, 1, 256, 256])
                    pred_masks_high_res: <class 'torch.Tensor'> torch.Size([1, 1, 1024, 1024])
                    obj_ptr: <class 'torch.Tensor'> torch.Size([1, 256])
                    maskmem_features: <class 'NoneType'> UNDEFINED_FORMAT
                    maskmem_pos_enc: <class 'NoneType'> UNDEFINED_FORMAT
            non_cond_frame_outputs:
        '''

        # Upscale pred masks to get high-res masks
        high_res_masks = torch.nn.functional.interpolate(
            pred_masks.to(self.predictor.device, non_blocking=True),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
        # ENCODE QUERY IMAGE INTO MEMORY ============================================================
        maskmem_features, maskmem_pos_enc = self.predictor._encode_new_memory(
            current_vision_feats=query_vision_features,
            feat_sizes=query_feature_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=True, # Because we are using the GT mask as input
        )
        # optionally offload the output to CPU memory to save GPU space
        maskmem_features = maskmem_features.to(torch.bfloat16).to(self.predictor.device, non_blocking=True)
        # Expand the maskmem_pos_enc to the actual batch size
        if maskmem_pos_enc is not None:
            maskmem_pos_enc = [x[0:1].clone() for x in maskmem_pos_enc]
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        
        # Add encoded mask memory features and posenc to the compact output
        output_dict['cond_frame_outputs'][0]["maskmem_features"] = maskmem_features
        output_dict['cond_frame_outputs'][0]["maskmem_pos_enc"] = expanded_maskmem_pos_enc
        
        # VISUALISATION OF THE PREDICTED QUERY MASK ============================================================
        # Resize mask to the original video resolution
        # any_res_masks = compact_current_out["pred_masks"].to(self.predictor.device, non_blocking=True)
        # if any_res_masks.shape[-2:] == (video_height, video_width):
        #     query_video_res_masks = any_res_masks
        # else:
        #     query_video_res_masks = torch.nn.functional.interpolate(
        #         any_res_masks,
        #         size=(video_height, video_width),
        #         mode="bilinear",
        #         align_corners=False,
        #     )
        # if self.predictor.non_overlap_masks:
        #     query_video_res_masks = self.predictor._apply_non_overlapping_constraints(query_video_res_masks)
    
        
        # ==============================================================================================================================================
        # SECOND STAGE
        # ==============================================================================================================================================
        
        # MEMORY ATTENTION ============================================================
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        target_high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(target_vision_features[:-1], target_feature_sizes[:-1])
        ]
        
        # Memory attention step: attention of target visual features with query encoded memory features
        pix_feat_with_mem = self.predictor._prepare_memory_conditioned_features(
            frame_idx=1,
            # TODO: should this be True or False?. In SAM 2 it is True, because no previous frames were uncond,
            # But with False it works much better, because it is using the information from query mask, otherwise ignored.
            is_init_cond_frame=False, # Basically this toggles attention to previous frames (ie query image) or not.
            current_vision_feats=target_vision_features[-1:],
            current_vision_pos_embeds=target_vision_pos_embeds[-1:],
            feat_sizes=target_feature_sizes[-1:],
            output_dict=output_dict,
            num_frames=2,  # TODO: This should probably be 2 (but is it 1?)
            track_in_reverse=False,
        )
    

        # Scale point to image size
        target_point_inputs['point_coords'] = target_point_inputs['point_coords'] * self.image_size
        
        target_multimask_output = self.predictor._use_multimask(is_init_cond_frame=False, point_inputs=target_point_inputs)
        # Expand pix_feat_with_mem to match the batch size
        pix_feat_with_mem_expanded = pix_feat_with_mem.expand(num_points, -1, -1, -1)
        
        if self.disable_custom_iou_embed:
            my_iou_token = None
        else:
            my_iou_token = self.iou_embed.weight.expand(num_points, -1, -1)
        
        # TARGET IMAGE MASK PREDICTION ============================================================
        # Using the target visual features after attending to the query memory features
        ( 
            target_low_res_multimasks,
            target_high_res_multimasks,
            my_iou_token_out, # my iou token out, used for passing into my MLP
            target_ious, # original predicted ious
            target_low_res_masks,
            target_high_res_masks,
            target_obj_ptr,
            target_object_score_logits # output of the occlusion MLP
        ) = self.predictor._forward_sam_heads(
                backbone_features=pix_feat_with_mem_expanded,
                point_inputs=target_point_inputs,
                mask_inputs=None,
                high_res_features=target_high_res_features,
                multimask_output=target_multimask_output,
                return_iou_token_out=True,
                merge_sparse_with_my_token=my_iou_token,
                disable_custom_iou_embed=self.disable_custom_iou_embed,
                disable_mlp_obj_scores=True,
        )
        
        target_pred_masks = self._fill_holes(target_low_res_masks)

        print("SUM my iou token out: ", my_iou_token_out.sum())
        my_iou_pred = self.iou_prediction_head(my_iou_token_out)
        
        return target_pred_masks, my_iou_pred