import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import cv2
import argparse
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from coco_dataloader import CocoQueryTargetDataset
from torchvision import transforms
import argparse
from sam2.utils.misc import load_video_frames
import pycocotools.mask as mask_utils
from sam2.utils.misc import fill_holes_in_mask_scores, concat_points
from sam2.modeling.sam2_utils import MLP
import wandb
import os
import random
from PIL import Image

def get_args_parser():
    parser = argparse.ArgumentParser("Finetune MLP")
    parser.add_argument("--model", type=str, required=True, help="Model size in ['tiny', 'small', 'base_plus', 'large']")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--wandb", type=str, help="W&B experiment name")
    parser.add_argument("--split", type=str, default="train2017", help="Split to use for training")
    
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
    for batch_item in batch:
        target, query = batch_item
    return target, query

def find_closest_mask_2(binary_masks, point):
    """ Find the closest mask to the point, and return the mask """
    
    closest_mask = None
    min_distance = float('inf')
    
    for mask in binary_masks:
        # Find all non-zero points in the mask
        mask_points = np.argwhere(mask)
        
        if len(mask_points) > 0:
            # Calculate distances from the given point to all points in the mask
            distances = np.sqrt(np.sum((mask_points - point)**2, axis=1))
            
            # Find the minimum distance for this mask
            min_mask_distance = np.min(distances)
            
            # Update the closest mask if this one is closer
            if min_mask_distance < min_distance:
                min_distance = min_mask_distance
                closest_mask = mask
    
    return closest_mask

def find_closest_mask(binary_masks, point):
    """ Find the closest mask to the point, and return the mask """
    
    closest_mask = None
    min_distance = float('inf')
    
    for mask in binary_masks:
        # Find all distances any point to the mask
        dist_transform = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
        # Select the distance at the point
        distance = dist_transform[point[0], point[1]]
        
        if distance < min_distance:
            min_distance = distance
            closest_mask = mask
    
    return closest_mask


def sample_point(binary_mask):
    # Sample a point from the binary mask
    # Return the coordinates of the point
    # If the binary mask is empty, return None
    if binary_mask.sum() == 0:
        raise ValueError("Binary mask is empty")
    else:
        points = np.argwhere(binary_mask)
        idx = np.random.choice(points.shape[0])
        return points[idx]

def sample_points(coco, target_img_info, target_anns, target_bad_anns, num_points=32, ratio_pos_neg=0.5):
    
    # Create a binary map of all the target_bad_anns joined
    bad_polys = [poly['segmentation'] for poly in coco.loadAnns(target_bad_anns)]
    bad_segms = [convert_polygon_to_mask(poly, width=target_img_info['width'], height=target_img_info['height']) for poly in bad_polys]
    bad_anns_binary = np.logical_or.reduce(bad_segms)
    
    # Convert anns to binary masks
    ann_masks = [convert_polygon_to_mask(ann['segmentation'], width=target_img_info['width'], height=target_img_info['height']) for ann in target_anns]
    # Create negative binary mask for sampling negative points: negative_mask = !bad_anns_binary & !ann_masks
    negative_binary_mask = np.logical_and(np.logical_not(bad_anns_binary), np.logical_not(np.logical_or.reduce(ann_masks)))
    
    points = []
    closest_anns = []
    for i in range(num_points):
        if np.random.rand() < ratio_pos_neg: # Positive point
            # Randomly select one annotation from the target_anns
            ann_idx = np.random.choice(len(ann_masks))
            ann_mask = ann_masks[ann_idx]
            point = sample_point(ann_mask)
        else: # Negative point
            point = sample_point(negative_binary_mask)
        # In both cases, find the closest ann to the point
        points.append(point)
        closest_anns.append(find_closest_mask(ann_masks, point))
        
    return points, closest_anns


def compute_iou(pred_mask, gt_mask):
    """ Compute the IOU between the predicted mask and the ground truth mask """   
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    return np.sum(intersection) / np.sum(union)

def main(args):
    # Set random seed for reproducibility
    seed = 22  # You can choose any integer value
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    split = args.split

    # Load the SAM2 Hiera model
    model_cfg = get_model_cfg(args.model)
    checkpoint = f"./checkpoints/sam2_hiera_{args.model}.pt"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    # Dataloader
    train_dataset = CocoQueryTargetDataset(root=f'./data/{split}',
                                json=f'./data/annotations/instances_{split}_tiny_filtered_by_0.6.json',
                                image_size=predictor.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        
    for step, (target, query) in enumerate(train_loader):
        
        # images, heights, widths, annotations, img_info
        
        # Unpack. Batch size is 1, atm
        target_img, target_img_info, target_anns, target_bad_anns = target
        query_img, query_img_info, query_anns, query_bad_anns = query
        
        video_height = target_img_info['height']
        video_width = target_img_info['width']
        
        # Add batch dimension to the target and query images
        target_img = target_img.unsqueeze(0)
        query_img = query_img.unsqueeze(0)
        
        # ENCODE QUERY and TARGET IMAGE with image encoder
        query_backbone_out = predictor.forward_image(query_img.to(predictor.device)) # Backbone output has keys: 'backbone_fpn', 'vision_pos_enc', 'vision_features'
        _, query_vision_features, query_vision_pos_embeds, query_feature_sizes = predictor._prepare_backbone_features(query_backbone_out)
        
        # TARGET IMAGE ENCODE ============================================================
        target_backbone_out = predictor.forward_image(target_img.to(predictor.device)) # Backbone output has keys: 'backbone_fpn', 'vision_pos_enc', 'vision_features'
        _, target_vision_features, target_vision_pos_embeds, target_feature_sizes = predictor._prepare_backbone_features(target_backbone_out)
        
        
        # ==============================================================================================================================================
        # FIRST STAGE: Forward pass for query image
        # ==============================================================================================================================================

        # QUERY IMAGE MASK PREPROCESSING ============================================================
        # Randomly select one annotation from the query image
        q_ann_idx = np.random.choice(len(query_anns))
        query_binary_mask = convert_polygon_to_mask(query_anns[q_ann_idx]['segmentation'], width=query_img_info['width'], height=query_img_info['height'])
        # Resize the binary mask to match the video dimensions
        resized_mask = torch.nn.functional.interpolate(
            torch.from_numpy(query_binary_mask).float().unsqueeze(0).unsqueeze(0),
            size=(video_height, video_width),
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
            current_vision_feats=query_vision_features,
            current_vision_pos_embeds=query_vision_pos_embeds,
            feat_sizes=query_feature_sizes,
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
            current_vision_feats=query_vision_features,
            feat_sizes=query_feature_sizes,
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
        if any_res_masks.shape[-2:] == (video_height, video_width):
            query_video_res_masks = any_res_masks
        else:
            query_video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_height, video_width),
                mode="bilinear",
                align_corners=False,
            )
        if predictor.non_overlap_masks:
            query_video_res_masks = predictor._apply_non_overlapping_constraints(query_video_res_masks)
    
        
        # ==============================================================================================================================================
        # SECOND STAGE
        # ==============================================================================================================================================
        
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
            is_init_cond_frame=True,
            current_vision_feats=target_vision_features[-1:],
            current_vision_pos_embeds=target_vision_pos_embeds[-1:],
            feat_sizes=target_feature_sizes[-1:],
            output_dict=output_dict,
            num_frames=2,
            track_in_reverse=False,
        )
        
        # SAMPLE NEGATIVE AND POSITIVE POINTS IN THE TARGET IMAGE ================================
    
        points, closest_anns = sample_points(train_dataset.coco, target_img_info, target_anns, target_bad_anns, num_points=32, ratio_pos_neg=0.8)

        for raw_point, closest_ann in zip(points, closest_anns):
            raw_label = 1
            
            raw_point = torch.tensor([raw_point], dtype=torch.float32).unsqueeze(0) # It should be a tensor of shape (1, 1, 2). add batch dimension
            raw_label = torch.tensor([raw_label], dtype=torch.int32).unsqueeze(0) # It should be a tensor of shape (1, 1). add batch dimension
            
            # Normalize the point to the video resolution
            point = raw_point / torch.tensor([video_width, video_height]).to(raw_point.device)
            # scale the (normalized) coordinates by the model's internal image size
            point = point * predictor.image_size
            point = point.to(predictor.device)
            label = raw_label.to(predictor.device)
            
            # Here is where we add the multiple points for the target image
            target_point_inputs = concat_points(None, point, label)
            # target_point_inputs = None
            target_multimask_output = predictor._use_multimask(is_init_cond_frame=False, point_inputs=target_point_inputs)
            
            # TARGET IMAGE MASK PREDICTION ============================================================
            # Using the target visual features after attending to the query memory features
            ( 
                target_low_res_multimasks,
                target_high_res_multimasks,
                target_ious, # original predicted ious
                target_low_res_masks,
                target_high_res_masks,
                target_obj_ptr,
                target_object_score_logits # output of the occlusion MLP
            ) = predictor._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=target_point_inputs,
                    mask_inputs=None,
                    high_res_features=target_high_res_features,
                    multimask_output=target_multimask_output,
                    disable_mlp_obj_scores=True
            )
            
            print("=====> IMPORTANT: target_object_score_logits: ", target_object_score_logits)

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
            if target_any_res_masks.shape[-2:] == (video_height, video_width):
                target_video_res_masks = target_any_res_masks
            else:
                target_video_res_masks = torch.nn.functional.interpolate(
                    target_any_res_masks,
                    size=(video_height, video_width),
                    mode="bilinear",
                    align_corners=False,
                )
            if predictor.non_overlap_masks:
                target_video_res_masks = predictor._apply_non_overlapping_constraints(target_video_res_masks)

            target_video_res_masks_binary = (target_video_res_masks > 0.0).detach().squeeze().cpu().numpy()
            

        
            # ==================================================================================================================        
            # ORIGINAL SAM 2 CODE
            # ==================================================================================================================        
            # Everything below is the original SAM2, just for comparison and checking that the values should be exactly the same
            
            print("\033[95m", "============================================================", "\033[0m")
            print("\033[95m", "================ SAM 2 ORIGINAL OUTPUTS ====================", "\033[0m")
            print("\033[95m", "============================================================", "\033[0m")
            
            inference_state = predictor.init_state(video_path=None, img_paths=[f'./data/{split}/{query_img_info["file_name"]}', f'./data/{split}/{target_img_info["file_name"]}'])        
            # expanded_img, pred_backbone_out, pred_vision_features, pred_vision_pos_embeds, pred_feature_sizes = predictor._get_image_feature(inference_state, 0, args.batch_size)
            # Add new mask to inference state
            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=0,
                mask=mask_inputs.squeeze(),
            )
            
            # CHECK 1: Mask predicted is the same in both cases
            mask1 = out_mask_logits[0].squeeze().cpu().numpy()
            mask2 = query_video_res_masks[0].squeeze().cpu().numpy()
            # print("\033[92m" + "CHECK 1: QUERY MASK predicted is the same in both cases" + "\033[0m")
            # print(mask1.shape, mask2.shape)
            if mask1.sum() == mask2.sum():
                print("\033[92m" + f"CHECK 1 PASSED: QUERY MASK predicted is the same in both cases (sum: {mask1.sum()}, {mask2.sum()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 1 FAILED: QUERY MASK predicted is different in both cases (sum: {mask1.sum()}, {mask2.sum()})" + "\033[0m")
            
            
            print("add new point")
            # Add point to frame 1 to simulate the target image prompt
            _, out_obj_ids2, out_mask_logits2 = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=1,
                obj_id=0,
                points=raw_point,
                labels=raw_label
            )
            
            
            print("propagate in video")
            # Propagate to target image
            video_segments = {}
            intermediate_sums = []
            for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
                intermediate_sums.append(masks[0].squeeze().cpu().numpy().sum())
                video_segments[frame_idx] = {
                    obj_id: (masks[j] > 0.0).squeeze().cpu().numpy()
                    for j, obj_id in enumerate(object_ids)
                }

            
            if intermediate_sums[0] == mask2.sum():
                print("\033[92m" + f"CHECK 2 PASSED: INTERMEDIATE QUERY MASK predicted is still the same (sum: {intermediate_sums[0]}, {mask2.sum()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 2 FAILED: INTERMEDIATE QUERY MASK predicted is different (sum: {intermediate_sums[0]}, {mask2.sum()})" + "\033[0m")
            
            frame_idx, masks = list(video_segments.items())[0] # Select the first frame
            if masks[0].squeeze().sum() == (mask2 > 0.0).sum():
                print("\033[92m" + f"CHECK 3 PASSED: BINARY QUERY MASK predicted is still the same (sum: {masks[0].squeeze().sum()}, {(mask2 > 0.0).sum()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 3 FAILED: BINARY QUERY MASK predicted is different (sum: {masks[0].squeeze().sum()}, {(mask2 > 0.0).sum()})" + "\033[0m")
            
            
            # CHECK 2: Mask predicted is the same in both cases
            frame_idx, masks = list(video_segments.items())[-1] # Select the last frame
            # Convert our predicted logits mask to binary mask
            target_video_res_masks_binary = (target_video_res_masks[0] > 0.0).detach().cpu().numpy()
            assert masks[0].squeeze().shape == target_video_res_masks_binary.squeeze().shape            
            
            if masks[0].squeeze().sum() == target_video_res_masks_binary.squeeze().sum():
                print("\033[92m" + f"CHECK 4 PASSED: TARGET BINARY MASK predicted is the same in both cases (sum: {masks[0].squeeze().sum()}, {target_video_res_masks_binary.squeeze().sum()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 4 FAILED: TARGET BINARY MASK predicted is different in both cases (sum: {masks[0].squeeze().sum()}, {target_video_res_masks_binary.squeeze().sum()})" + "\033[0m")
                
            print("out_mask_logits2[0].squeeze().cpu().numpy().sum(): ", out_mask_logits2[0].squeeze().cpu().numpy().sum())
            print("target_video_res_masks.squeeze().sum().item(): ", target_video_res_masks.squeeze().sum().item())
            print("intermediate_sums[-1]: ", intermediate_sums[-1])
                
            if (out_mask_logits2[0].squeeze().cpu().numpy().sum() == target_video_res_masks.squeeze().sum().item()):
                print("\033[92m" + f"CHECK 5 PASSED: TARGET LOGITS predicted is the same in both cases (sum: {out_mask_logits2[0].squeeze().cpu().numpy().sum()}, {target_video_res_masks.squeeze().sum().item()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 5 FAILED: TARGET LOGITS predicted is different in both cases (sum: {out_mask_logits2[0].squeeze().cpu().numpy().sum()}, {target_video_res_masks.squeeze().sum().item()})" + "\033[0m")


            print("\033[93m" + "========== (additional checks) =================" + "\033[0m")
            if output_dict['cond_frame_outputs'][0]['maskmem_features'].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_features'].sum().item():
                print("\033[92m" + f"CHECK 6.1 PASSED: MASKMEM FEATURES predicted is the same in both cases (sum: {output_dict['cond_frame_outputs'][0]['maskmem_features'].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_features'].sum().item()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 6.1 FAILED: MASKMEM FEATURES predicted is different in both cases (sum: {output_dict['cond_frame_outputs'][0]['maskmem_features'].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_features'].sum().item()})" + "\033[0m")
            # TODO: CHECK WHY POSENC IS SLIGHTLY DIFFERENT (just some last decimals, 127058.875 vs 127058.890625)?
            if (output_dict['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item()):
                print("\033[92m" + f"CHECK 6.2 PASSED: MASKMEM POSENC predicted is the same in both cases (sum: {output_dict['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 6.2 FAILED: MASKMEM POSENC predicted is different in both cases (sum: {output_dict['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['maskmem_pos_enc'][0].sum().item()})" + "\033[0m")
            if (output_dict['cond_frame_outputs'][0]['pred_masks'].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['pred_masks'].sum().item()):
                print("\033[92m" + f"CHECK 6.3 PASSED: PRED MASK predicted is the same in both cases (sum: {output_dict['cond_frame_outputs'][0]['pred_masks'].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['pred_masks'].sum().item()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 6.3 FAILED: PRED MASK predicted is different in both cases (sum: {output_dict['cond_frame_outputs'][0]['pred_masks'].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['pred_masks'].sum().item()})" + "\033[0m")
            if (output_dict['cond_frame_outputs'][0]['obj_ptr'].sum().item() == inference_state['output_dict']['cond_frame_outputs'][0]['obj_ptr'].sum().item()):
                print("\033[92m" + f"CHECK 6.4 PASSED: OBJ PTR predicted is the same in both cases (sum: {output_dict['cond_frame_outputs'][0]['obj_ptr'].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['obj_ptr'].sum().item()})" + "\033[0m")
            else:
                print("\033[91m" + f"CHECK 3.4 FAILED: OBJ PTR predicted is different in both cases (sum: {output_dict['cond_frame_outputs'][0]['obj_ptr'].sum().item()}, {inference_state['output_dict']['cond_frame_outputs'][0]['obj_ptr'].sum().item()})" + "\033[0m")
        
        
            # Below, we plot all the masks, that is:
            #  - the query ground truth and the predicted query mask using our method
            #  - the query ground truth and the predicted query mask using the original SAM
            #  - the target ground truth, the target point (as a star), and the predicted target mask using our method
            #  - the target ground truth, the target point (as a star), and the predicted target mask using the original SAM
            
            # QUERY MASKS
            query_real_gt_mask = query_binary_mask
            query_shape = query_real_gt_mask.shape
            
            my_query_pred_logits = nn.functional.interpolate(query_video_res_masks,
                                                                    size=query_shape,
                                                                    mode="bilinear",
                                                                    align_corners=False).squeeze().cpu().numpy()
            sam_query_pred_logits = nn.functional.interpolate(out_mask_logits,
                                                                    size=query_shape,
                                                                    mode="bilinear",
                                                                    align_corners=False).squeeze().cpu().numpy()

            # TARGET MASKS
            target_real_gt_mask = closest_ann

            my_target_pred_logits = target_video_res_masks[0].detach().squeeze().cpu().numpy()
            sam_target_pred_logits = out_mask_logits2[0].squeeze().cpu().numpy()
            
            pil_query_img = Image.open(f'./data/{split}/{query_img_info["file_name"]}')
            pil_target_img = Image.open(f'./data/{split}/{target_img_info["file_name"]}')
            
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2, figsize=(20, 20))

            # Query image with ground truth and our prediction
            axs[0, 0].imshow(pil_query_img)
            axs[0, 0].imshow(my_query_pred_logits, cmap='jet', alpha=0.5)
            axs[0, 0].imshow(query_real_gt_mask, cmap='jet', alpha=0.5)
            axs[0, 0].set_title('Query: Ground Truth & Our Prediction')

            # Query image with ground truth and SAM prediction
            axs[0, 1].imshow(pil_query_img)
            axs[0, 1].imshow(sam_query_pred_logits, cmap='jet', alpha=0.5)
            axs[0, 1].imshow(query_binary_mask, cmap='jet', alpha=0.5)
            axs[0, 1].set_title('Query: Ground Truth & SAM Prediction')

            # Target image with ground truth, point, and our prediction
            axs[1, 0].imshow(pil_target_img)
            axs[1, 0].imshow(closest_ann, cmap='jet', alpha=0.5)
            axs[1, 0].imshow(my_target_pred_logits, cmap='jet', alpha=0.5)
            axs[1, 0].plot(raw_point.squeeze().cpu().numpy()[1], raw_point.squeeze().cpu().numpy()[0], 'r*', markersize=20)
            axs[1, 0].set_title('Target: Ground Truth, Point & Our Prediction')

            # Target image with ground truth, point, and SAM prediction
            axs[1, 1].imshow(pil_target_img)
            axs[1, 1].imshow(closest_ann, cmap='jet', alpha=0.5)
            axs[1, 1].imshow(sam_target_pred_logits, cmap='jet', alpha=0.5)
            axs[1, 1].plot(raw_point.squeeze().cpu().numpy()[1], raw_point.squeeze().cpu().numpy()[0], 'r*', markersize=20)
            axs[1, 1].set_title('Target: Ground Truth, Point & SAM Prediction')

            plt.tight_layout()
            plt.savefig(f"./plots/plot_{step}.png")
            plt.close()
            

            exit()
        

    
    
if __name__ == "__main__":
    
    args = get_args_parser()
    main(args)
