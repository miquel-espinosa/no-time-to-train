import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from coco_dataloader import CocoDataset
from torchvision import transforms
import argparse
from sam2.utils.misc import load_video_frames
import pycocotools.mask as mask_utils
from sam2.utils.misc import fill_holes_in_mask_scores

def get_args_parser():
    parser = argparse.ArgumentParser("Finetune MLP")
    parser.add_argument("--model", type=str, required=True, help="Model size in ['tiny', 'small', 'base_plus', 'large']")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    return parser.parse_args()


def get_model_cfg(model_name):
    if model_name == 'base_plus':
        model_cfg = "sam2_hiera_b+.yaml"
    else:
        model_cfg = f"sam2_hiera_{model_name[0]}.yaml"
    return model_cfg


def convert_polygon_to_mask(polygon, width, height):
    rle = mask_utils.frPyObjects(polygon, height, width)
    binary_mask = mask_utils.decode(rle).squeeze()
    # If there are more than one channel, merge them into a single channel.
    # This means that if we have multiple polygons for the same annotation, 
    # we merge them into a single mask.
    if binary_mask.ndim == 3:
        binary_mask = binary_mask.max(axis=2)
    return binary_mask


def collate_fn(batch):
    images = []
    heights = []
    widths = []
    annotations = []
    img_infos = []
    for image, height, width, annotation, img_info in batch:
        images.append(image)
        heights.append(height)
        widths.append(width)
        annotations.append(annotation)
        img_infos.append(img_info)
        
    images = torch.stack(images)
    return images, heights, widths, annotations, img_infos

def main(args):

    split = 'val2017'

    # Load the SAM2 Hiera model
    model_cfg = get_model_cfg(args.model)
    checkpoint = f"./checkpoints/sam2_hiera_{args.model}.pt"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    # Dataloader
    train_dataset = CocoDataset(root=f'./data/{split}',
                                json=f'./data/annotations/instances_{split}.json',
                                image_size=predictor.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)

    for images, heights, widths, annotations, img_info in train_loader:
        
        images = images.to(predictor.device) # To float, to cuda. They already include the batch dimension     
        target_image = images.clone() # Dummy target image
        
        # ==============================================================================================================================================
        # FIRST STAGE
        # ==============================================================================================================================================
        
        # ENCODE QUERY IMAGE with image encoder
        backbone_out = predictor.forward_image(images) # Backbone output has keys: 'backbone_fpn', 'vision_pos_enc', 'vision_features'
        _, vision_features, vision_pos_embeds, feature_sizes = predictor._prepare_backbone_features(backbone_out)        

        # QUERY IMAGE MASK PREPROCESSING ============================================================
        binary_mask = convert_polygon_to_mask(annotations[0]['segmentation'], heights[0], widths[0])
        # Resize the binary mask to match the video dimensions
        resized_mask = torch.nn.functional.interpolate(
            torch.from_numpy(binary_mask).float().unsqueeze(0).unsqueeze(0),
            size=(heights[0], widths[0]),
            mode='nearest'
        ).squeeze().numpy()
        # Convert the resized binary mask to the format expected by SAM2
        sam_mask = torch.from_numpy(resized_mask).bool()
        
        mask_H, mask_W = sam_mask.shape
        mask_inputs_orig = sam_mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(predictor.device)
        
        # resize the mask if it doesn't match the model's image size
        if mask_H != predictor.image_size or mask_W != predictor.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(predictor.image_size, predictor.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig
        
        # PREPARE OUTPUT DICT ====================================================================
        # Create output_dict for storing needed values
        output_dict = {}
        output_dict['cond_frame_outputs'] = {}
        output_dict['non_cond_frame_outputs'] = {}
        
        # QUERY IMAGE MASK PREDICTION ============================================================
        current_out = predictor.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_features,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feature_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=1,
            track_in_reverse=False,
            run_mem_encoder=False, # We will encode the mask later
            prev_sam_mask_logits=None,
        )
        # Store the current output in the output_dict
        output_dict['cond_frame_outputs'][0] = current_out
        
        pred_masks_gpu = current_out["pred_masks"]
        # fill holes in mask
        if predictor.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, predictor.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(predictor.device, non_blocking=True)

        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "pred_masks": pred_masks,
            "obj_ptr": current_out["obj_ptr"],
        } 
        
        high_res_masks = torch.nn.functional.interpolate(
            compact_current_out["pred_masks"].to(predictor.device, non_blocking=True),
            size=(predictor.image_size, predictor.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
        # ENCODE QUERY IMAGE INTO MEMORY ============================================================
        maskmem_features, maskmem_pos_enc = predictor._encode_new_memory(
            current_vision_feats=vision_features,
            feat_sizes=feature_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=True, # Because we are using the GT mask as input
        )

        # optionally offload the output to CPU memory to save GPU space
        maskmem_features = maskmem_features.to(torch.bfloat16).to(predictor.device, non_blocking=True)
        # Expand the maskmem_pos_enc to the actual batch size
        if maskmem_pos_enc is not None:
            maskmem_pos_enc = [x[0:1].clone() for x in maskmem_pos_enc]
            expanded_maskmem_pos_enc = [
                x.expand(args.batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        
        # Add encoded mask memory features and posenc to the compact output
        compact_current_out["maskmem_features"] = maskmem_features
        compact_current_out["maskmem_pos_enc"] = expanded_maskmem_pos_enc
        # Update our output_dict with the compact output
        output_dict['cond_frame_outputs'][0] = compact_current_out
        
        # VISUALISATION OF THE PREDICTED QUERY MASK ============================================================
        # Resize mask to the original video resolution
        any_res_masks = compact_current_out["pred_masks"].to(predictor.device, non_blocking=True)
        if any_res_masks.shape[-2:] == (heights[0], widths[0]):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(heights[0], widths[0]),
                mode="bilinear",
                align_corners=False,
            )
        if predictor.non_overlap_masks:
            video_res_masks = predictor._apply_non_overlapping_constraints(video_res_masks)
       
        
        # ==============================================================================================================================================
        # SECOND STAGE
        # ==============================================================================================================================================
        
        # TARGET IMAGE ENCODE ============================================================
        target_backbone_out = predictor.forward_image(target_image) # Backbone output has keys: 'backbone_fpn', 'vision_pos_enc', 'vision_features'
        _, target_vision_features, target_vision_pos_embeds, target_feature_sizes = predictor._prepare_backbone_features(target_backbone_out)
        
        
        # MEMORY ATTENTION ============================================================
        target_current_out = {}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        target_high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(target_vision_features[:-1], target_feature_sizes[:-1])
        ]
        # Memory attention step: attention of target visual features with query encoded memory features
        pix_feat_with_mem = predictor._prepare_memory_conditioned_features(
            frame_idx=1,
            is_init_cond_frame=False,
            current_vision_feats=target_vision_features[-1:],
            current_vision_pos_embeds=target_vision_pos_embeds[-1:],
            feat_sizes=target_feature_sizes[-1:],
            output_dict=output_dict,
            num_frames=1,
            track_in_reverse=False,
        )
        
        # Here is where we need to add the multiple points for the target image
        target_point_inputs = None
        target_multimask_output = predictor._use_multimask(is_init_cond_frame=False, point_inputs=target_point_inputs)
        
        # TARGET IMAGE MASK PREDICTION ============================================================
        # Using the target visual features after attending to the query memory features
        _, _, _, target_low_res_masks, target_high_res_masks, target_obj_ptr, _ = predictor._forward_sam_heads(
            backbone_features=pix_feat_with_mem,
            point_inputs=target_point_inputs,
            mask_inputs=None,
            high_res_features=target_high_res_features,
            multimask_output=target_multimask_output,
        )

        target_current_out["pred_masks"] = target_low_res_masks
        target_current_out["pred_masks_high_res"] = target_high_res_masks
        target_current_out["obj_ptr"] = target_obj_ptr
        
        target_pred_masks_gpu = target_current_out["pred_masks"]
                
        if predictor.fill_hole_area > 0:
            target_pred_masks_gpu = fill_holes_in_mask_scores(
                target_pred_masks_gpu, predictor.fill_hole_area
            )
        target_pred_masks = target_pred_masks_gpu.to(predictor.device, non_blocking=True)
                
        # Resize predicted target mask to the original video resolution
        target_any_res_masks = target_pred_masks.to(predictor.device, non_blocking=True)
        if target_any_res_masks.shape[-2:] == (heights[0], widths[0]):
            target_video_res_masks = target_any_res_masks
        else:
            target_video_res_masks = torch.nn.functional.interpolate(
                target_any_res_masks,
                size=(heights[0], widths[0]),
                mode="bilinear",
                align_corners=False,
            )
        if predictor.non_overlap_masks:
            target_video_res_masks = predictor._apply_non_overlapping_constraints(target_video_res_masks)

        
        # ==================================================================================================================        
        # ORIGINAL SAM 2 CODE
        # ==================================================================================================================        
        # Everything below is the original SAM2, just for comparison and checking that the values should be exactly the same
        
        inference_state = predictor.init_state(video_path=None, img_paths=[f'./data/{split}/{img_info[0]["file_name"]}', f'./data/{split}/{img_info[0]["file_name"]}'])        
        expanded_img, pred_backbone_out, pred_vision_features, pred_vision_pos_embeds, pred_feature_sizes = predictor._get_image_feature(inference_state, 0, args.batch_size)
        # Add new mask to inference state
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=0,
            mask=mask_inputs.squeeze(),
        )
        video_segments = {}
        for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
            video_segments[frame_idx] = {
                obj_id: (masks[j] > 0.0).squeeze().cpu().numpy()
                for j, obj_id in enumerate(object_ids)
            }
        
        # CHECK 1: Mask predicted is the same in both cases
        mask1 = out_mask_logits[0].squeeze().cpu().numpy()
        mask2 = video_res_masks[0].squeeze().cpu().numpy()
        print("\033[92m" + "CHECK 1: Mask predicted is the same in both cases" + "\033[0m")
        print(mask1.shape, mask2.shape)
        print(mask1.sum(), mask2.sum())
        print(mask1 == mask2)
        
        # CHECK 2: Mask predicted is the same in both cases
        frame_idx, masks = list(video_segments.items())[-1] # Select the last frame
        # Convert our predicted logits mask to binary mask
        target_video_res_masks_binary = (target_video_res_masks[0] > 0.0).detach().cpu().numpy()
        print("\033[92m" + "CHECK 2: Mask predicted is the same in both cases" + "\033[0m")
        print(masks[0].squeeze().sum() == target_video_res_masks_binary.squeeze().sum())
        print(masks[0].squeeze().sum(), target_video_res_masks_binary.squeeze().sum())

        # CHECK 3: Compare the values of the maskmem features, posenc, pred masks, obj ptr
        print("\033[92m" + "CHECK 3: Compare the values of the maskmem features, posenc, pred masks, obj ptr" + "\033[0m")
        print("maskmem features: ", output_dict['cond_frame_outputs'][0]['maskmem_features'].sum().item(), inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_features'].sum().item())
        print("maskmem features: ", output_dict['cond_frame_outputs'][0]['maskmem_features'].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_features'].sum().item())
        # TODO: CHECK WHY POSENC IS SLIGHTLY DIFFERENT (just some last decimals, 127058.875 vs 127058.890625)?
        print("maskmem posenc: ", output_dict['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item(), inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item())
        print("maskmem posenc: ", output_dict['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item())
        print("pred masks: ", output_dict['cond_frame_outputs'][0]['pred_masks'].sum().item(), inference_state['output_dict']['cond_frame_outputs'][0]['pred_masks'].sum().item())
        print("pred masks: ", output_dict['cond_frame_outputs'][0]['pred_masks'].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['pred_masks'].sum().item())
        print("obj ptr: ", output_dict['cond_frame_outputs'][0]['obj_ptr'].sum().item(), inference_state['output_dict']['cond_frame_outputs'][0]['obj_ptr'].sum().item())
        print("obj ptr: ", output_dict['cond_frame_outputs'][0]['obj_ptr'].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['obj_ptr'].sum().item())
        

        break
        

    
    
if __name__ == "__main__":
    
    args = get_args_parser()
    main(args)
