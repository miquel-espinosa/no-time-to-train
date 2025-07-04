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
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser("Finetune MLP")
    parser.add_argument("--model", type=str, required=True, help="Model size in ['tiny', 'small', 'base_plus', 'large']")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="Weight decay")
    parser.add_argument("--wandb", type=str, help="W&B experiment name")
    parser.add_argument("--split", type=str, default="train2017", help="Split to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--num_points", type=int, default=32, help="Number of points to sample")
    parser.add_argument("--ratio_pos_neg", type=float, default=0.5, help="Ratio of positive to negative points")
    parser.add_argument("--disable_custom_iou_embed", action="store_true", help="Disable custom iou embed")
    
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
    pos_neg = []
    for i in range(num_points):
        if np.random.rand() < ratio_pos_neg: # Positive point
            # Randomly select one annotation from the target_anns
            ann_idx = np.random.choice(len(ann_masks))
            ann_mask = ann_masks[ann_idx]
            if ann_mask.sum() == 0:
                try:
                    print("\033[93mWARNING: SKIPPING positive point because ann_mask is empty (no positive points to sample)\033[0m")
                    print(f"\033[93mThis should not happen. Check annotation ann_idx: {target_anns[ann_idx]['id']}, area: {target_anns[ann_idx]['area']}\033[0m")
                except:
                    continue
                continue
            point = sample_point(ann_mask)
            points.append(point)
            closest_anns.append(ann_mask)
            pos_neg.append(1)
        else: # Negative point
            if negative_binary_mask.sum() == 0:
                print("SKIPPING negative point because negative binary mask is empty (no negative points to sample)")
                continue
            point = sample_point(negative_binary_mask)
            points.append(point)
            closest_anns.append(find_closest_mask(ann_masks, point))
            pos_neg.append(0)
        
    return np.array(points), np.array(closest_anns), np.array(pos_neg)


def compute_ious(pred_masks, gt_masks):
    """ Compute the IOU between the predicted mask and the ground truth mask. Take into account batch dimension."""
    intersections = torch.logical_and(pred_masks, gt_masks)
    unions = torch.logical_or(pred_masks, gt_masks)
    return torch.sum(intersections, dim=(1, 2)) / torch.sum(unions, dim=(1, 2))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MyModel(nn.Module):
    def __init__(self, transformer_dim, iou_head_hidden_dim, num_mask_tokens,
                 iou_head_depth, iou_prediction_use_sigmoid, disable_custom_iou_embed):
        super(MyModel, self).__init__()
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, num_mask_tokens, iou_head_depth, sigmoid_output=iou_prediction_use_sigmoid
        )
        if not disable_custom_iou_embed:
            self.iou_embed = nn.Embedding(1, transformer_dim)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def plot_query_target_masks(query_img_info, target_img_info, query_video_res_masks,
                            gt_target_mask, target_video_res_masks, raw_point,
                            my_iou_pred, iou_gt, target_video_res_masks_binary, path_and_filename,
                            predictor_image_size, pos_neg, args):

    # Load images
    pil_query_img = Image.open(f'./data/{args.split}/{query_img_info["file_name"]}')
    pil_target_img = Image.open(f'./data/{args.split}/{target_img_info["file_name"]}')
    # Convert to numpy
    raw_point = raw_point.cpu().numpy()
    gt_target_mask = gt_target_mask.cpu().numpy()
    target_video_res_masks = target_video_res_masks.detach().cpu().numpy()
    target_video_res_masks_binary = target_video_res_masks_binary.cpu().numpy()

    # Prepare masks
    # query_pred_mask = query_video_res_masks[0].detach().squeeze().cpu().numpy()
    query_pred_mask = nn.functional.interpolate(query_video_res_masks, size=(pil_query_img.size[1], pil_query_img.size[0]),
                                                mode="bilinear", align_corners=False).detach().squeeze().cpu().numpy()
    target_pred_mask = target_video_res_masks[0]
    
    if pos_neg == 1:
        pos_neg_str = "Positive"
        color = "g"
    else:
        pos_neg_str = "Negative"
        color = "r"

    # Create figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # Plot query image and mask
    axs[0, 0].imshow(pil_query_img)
    axs[0, 0].imshow(query_pred_mask, cmap='jet', alpha=0.5)
    axs[0, 0].set_title(f"Query Image with Predicted Mask")
    axs[0, 0].axis('off')

    # Plot target image and predicted mask
    axs[0, 1].imshow(pil_target_img)
    axs[0, 1].imshow(target_pred_mask, cmap='jet', alpha=0.5)
    axs[0, 1].plot(raw_point[0, 0], raw_point[0, 1], f'{color}*', markersize=20, markeredgecolor='w')
    axs[0, 1].set_title(f"Target Image with Predicted Mask")
    axs[0, 1].axis('off')

    # Plot target image and binary predicted mask
    axs[1, 0].imshow(pil_target_img)
    axs[1, 0].imshow(gt_target_mask, cmap='jet', alpha=0.5)
    axs[1, 0].set_title(f"Target Image with Ground Truth Mask")
    axs[1, 0].axis('off')

    # Plot target image with point
    axs[1, 1].imshow(pil_target_img)
    axs[1, 1].imshow(target_video_res_masks_binary, cmap='jet', alpha=0.5)
    axs[1, 1].plot(raw_point[0, 0], raw_point[0, 1], f'{color}*', markersize=20, markeredgecolor='w')
    axs[1, 1].set_title(f"Target Image with {pos_neg_str} Point (IOU predicted: {my_iou_pred.detach().squeeze().cpu().numpy():.2f}, IOU ground truth: {iou_gt.squeeze():.2f})")
    axs[1, 1].axis('off')

    # Add colorbar
    for i in range(2):
        for j in range(2):
            if i != 1 or j != 1:  # Skip the last subplot (point plot)
                plt.colorbar(axs[i, j].images[1], ax=axs[i, j], fraction=0.046, pad=0.04)

    if args.wandb is not None:
        # Add title to the plot with IOU predicted and IOU ground truth, and experiment name
        title = f"{args.wandb} - IOU predicted: {my_iou_pred.detach().squeeze().cpu().numpy():.2f}, IOU ground truth: {iou_gt.squeeze():.2f}"
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(f"{path_and_filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    set_seed(args.seed)

    split = args.split

    # Load the SAM2 Hiera model
    model_cfg = get_model_cfg(args.model)
    checkpoint = f"./checkpoints/sam2_hiera_{args.model}.pt"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    if args.wandb is not None:
        wandb.init(project="mlp_finetune", name=args.wandb)
        saved_models_path = f'finetune/saved_models/{args.wandb}'
        saved_plots_path = f'finetune/plots/{args.wandb}'
        if not os.path.exists(saved_models_path):
            os.makedirs(saved_models_path)
        if not os.path.exists(saved_plots_path):
            os.makedirs(saved_plots_path)
    else:
        saved_plots_path = f'./plots/debug'
        if not os.path.exists(saved_plots_path):
            os.makedirs(saved_plots_path)
        # Print warning in red
        print("\033[91mWarning: No experiment name provided, checkpoints will not be saved, and results will not be logged to W&B.\033[0m")
        print(f"\033[91mSaving plots to {saved_plots_path}\033[0m")
    
    # Freeze all paramters
    for param in predictor.parameters():
        param.requires_grad = False
        
    print("Number of trainable parameters in SAM 2: ", count_parameters(predictor))
   
    mymodel = MyModel(
        predictor.sam_mask_decoder.transformer_dim,
        predictor.sam_mask_decoder.iou_head_hidden_dim,
        predictor.sam_mask_decoder.num_mask_tokens,
        predictor.sam_mask_decoder.iou_head_depth,
        predictor.sam_mask_decoder.iou_prediction_use_sigmoid,
        args.disable_custom_iou_embed,
    )
    mymodel.to(predictor.device)
    
    print("Number of trainable parameters in my MLP: ", count_parameters(mymodel))
    
    # Optimizer
    optimizer = torch.optim.AdamW(params=mymodel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    
    # Dataloader
    train_dataset = CocoQueryTargetDataset(root=f'./data/{split}',
                                json=f'./data/annotations/instances_{split}_tiny_filtered_by_0.6.json',
                                image_size=predictor.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        
        for step, (target, query) in enumerate(tqdm(train_loader)):
            
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
            # SAMPLE NEGATIVE AND POSITIVE POINTS IN THE TARGET IMAGE ================================
        
            raw_points, closest_anns, pos_neg = sample_points(train_dataset.coco, target_img_info, target_anns, target_bad_anns,
                                                 num_points=args.num_points, ratio_pos_neg=args.ratio_pos_neg)

            # Convert points to torch tensor
            raw_points = torch.tensor(raw_points, dtype=torch.float32).unsqueeze(1).to(predictor.device) # Shape (B, 1, 2)
            # 1 for positive points, 0 for negative points
            labels = torch.ones((raw_points.shape[0], raw_points.shape[1]), dtype=torch.int32).to(predictor.device) # Shape (B, 1)
            closest_anns = torch.tensor(closest_anns, dtype=torch.int32).to(predictor.device) # Shape (B, H, W)
            pos_neg = torch.tensor(pos_neg, dtype=torch.int32).to(predictor.device) # Shape (B, 1)
            
            # Flip all points  (b, x, y) -> (b, y, x) to match the format of the model
            raw_points = torch.flip(raw_points, dims=[2])
            
            # Normalize the points (B, X, Y) to the video resolution
            points = raw_points / torch.tensor([video_width, video_height]).to(raw_points.device)
            # Scale the (normalized) coordinates by the model's internal image size
            points = points * predictor.image_size

            target_point_inputs = {"point_coords": points, "point_labels": labels}
            target_multimask_output = predictor._use_multimask(is_init_cond_frame=False, point_inputs=target_point_inputs)

            # Expand pix_feat_with_mem to match the batch size
            pix_feat_with_mem_expanded = pix_feat_with_mem.expand(points.shape[0], -1, -1, -1)
            
            if args.disable_custom_iou_embed:
                my_iou_token = None
            else:
                my_iou_token = mymodel.iou_embed.weight.expand(points.shape[0], -1, -1)
            
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
            ) = predictor._forward_sam_heads(
                    backbone_features=pix_feat_with_mem_expanded,
                    point_inputs=target_point_inputs,
                    mask_inputs=None,
                    high_res_features=target_high_res_features,
                    multimask_output=target_multimask_output,
                    return_iou_token_out=True,
                    merge_sparse_with_my_token=my_iou_token,
                    disable_custom_iou_embed=args.disable_custom_iou_embed,
                    disable_mlp_obj_scores=True,
            )
            
            target_pred_masks_gpu = target_low_res_masks
                    
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
            
            # if predictor.non_overlap_masks:
            #     print("WE DONT REACH THIS")
            #     target_video_res_masks = predictor._apply_non_overlapping_constraints(target_video_res_masks)
            
            
            target_video_res_masks_binary = (target_video_res_masks > 0.0).squeeze()
                
            # IOU PREDICTION ============================================================
            my_iou_pred = mymodel.iou_prediction_head(my_iou_token_out)
            
            print("SUM my iou token out: ", my_iou_token_out.sum())

            # if target_multimask_output:
            #     my_iou_pred = my_iou_pred[:, 1:]
            # else:
            #     my_iou_pred = my_iou_pred[:, 0:1]
            
            # At the moment, we just use the first mask for the iou prediction
            my_iou_pred = my_iou_pred[:, 0:1].squeeze()
            # Compute real IOU between the predicted mask and the closest annotation
            iou = compute_ious(pred_masks=target_video_res_masks_binary, gt_masks=closest_anns)
            # Abs and mean to get a scalar
            iou_loss = torch.abs(my_iou_pred - iou).mean()
            print(f"IOU loss: {iou_loss}")

            # ---------------------------------------------------------------------------------------------------------------
            # Apply back propagation
            mymodel.zero_grad()
            scaler.scale(iou_loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            if step % 200 == 0:
                idx = np.random.choice(points.shape[0])
                # idx = 3
                # for idx in range(points.shape[0]):
                path_and_filename = f"{saved_plots_path}/pred_masks_epoch_{epoch}_step_{step}_point_{idx}.png"
                plot_query_target_masks(query_img_info, target_img_info, query_video_res_masks, closest_anns[idx],
                                        target_video_res_masks[idx], raw_points[idx], my_iou_pred[idx], iou[idx],
                                        target_video_res_masks_binary[idx], path_and_filename,
                                        predictor.image_size, pos_neg[idx], args)

            if step % 50 == 0:
                # print("iou_loss: ", iou_loss.item())

                if args.wandb is not None:
                    wandb.log({"iou_loss": iou_loss.item()}) #, "iou_pred": my_iou_pred.item(), "iou_gt": iou.item()})
        
        
            if args.wandb is not None:
                if step == 0 and epoch == 0:
                    torch.save(mymodel.state_dict(), f"{saved_models_path}/init_model.pth")

                if step % 500 == 0:
                    torch.save(mymodel.state_dict(), f"{saved_models_path}/iou_mlp_epoch_{epoch}_step_{step}.pth")
                    print(f"Saved model at epoch {epoch} step {step}")
        
        # if args.wandb is not None:
        #     torch.save(mymodel.state_dict(), f"{saved_models_path}/iou_mlp_epoch_{epoch}.pth")
        #     print(f"Saved model at epoch {epoch}")
        

    
    
if __name__ == "__main__":
    
    args = get_args_parser()
    main(args)
