import os
import random
import cv2
import argparse
import numpy as np
import argparse
import pycocotools.mask as mask_utils
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coco_dataloader import CocoQueryTargetDataset
from sam2_query_target import SAM2QueryTarget

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
    pil_query_img = Image.open(f'./data/coco/{args.split}/{query_img_info["file_name"]}')
    pil_target_img = Image.open(f'./data/coco/{args.split}/{target_img_info["file_name"]}')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the SAM2 Hiera model
    model_cfg = get_model_cfg(args.model)
    checkpoint = f"./checkpoints/sam2_hiera_{args.model}.pt"
    
    model = SAM2QueryTarget(model_cfg, checkpoint, disable_custom_iou_embed=args.disable_custom_iou_embed).to(device)
    
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
        
    print("Number of trainable parameters in SAM2QueryTarget: ", count_parameters(model))
    
    # Optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    
    # Dataloader
    train_dataset = CocoQueryTargetDataset(root=f'./data/coco/{split}',
                                json=f'./data/coco/annotations_refsam2/instances_{split}_tiny_filtered_by_0.6.json',
                                image_size=model.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    for epoch in range(args.epochs):
        
        for step, (target, query) in enumerate(tqdm(train_loader)):
                        
            # Unpack. Batch size is 1, atm
            target_img, target_img_info, target_anns, target_bad_anns = target
            query_img, query_img_info, query_anns, query_bad_anns = query
            
            video_height = target_img_info['height']
            video_width = target_img_info['width']
            
            # Add batch dimension to the target and query images
            target_img = target_img.unsqueeze(0)
            query_img = query_img.unsqueeze(0)
            
            # QUERY IMAGE MASK PREPROCESSING =======================================================
            # NOTE: This can go in the dataset class
            # Randomly select one annotation from the query image
            q_ann_idx = np.random.choice(len(query_anns))
            query_binary_mask = convert_polygon_to_mask(query_anns[q_ann_idx]['segmentation'],
                                                        width=query_img_info['width'],
                                                        height=query_img_info['height'])
            # Resize the binary mask to match the video dimensions
            query_resized_binary_mask = torch.nn.functional.interpolate(
                torch.from_numpy(query_binary_mask).float().unsqueeze(0).unsqueeze(0),
                size=(video_height, video_width),
                mode='nearest'
            ).squeeze().numpy()
            # Convert the resized binary mask to the format expected by SAM2
            query_resized_binary_mask = torch.from_numpy(query_resized_binary_mask).bool()
             # add batch and channel dimension
            query_resized_binary_mask = query_resized_binary_mask[None, None].float().to(device)
            # ======================================================================================
            
            
            # PREPARE POINTS FOR TARGET IMAGE ======================================================
            # NOTE: This part can also go into the dataset class, dataloading step
            target_points, target_closest_anns, target_pos_neg = sample_points(train_dataset.coco, target_img_info, target_anns, target_bad_anns,
                                                num_points=args.num_points, ratio_pos_neg=args.ratio_pos_neg)
            # Convert points to torch tensor
            target_points = torch.tensor(target_points, dtype=torch.float32).unsqueeze(1).to(device) # Shape (B, 1, 2)
            # 1 for positive points, 0 for negative points
            target_labels = torch.ones((target_points.shape[0], target_points.shape[1]), dtype=torch.int32).to(device) # Shape (B, 1)
            target_closest_anns = torch.tensor(target_closest_anns, dtype=torch.int32).to(device) # Shape (B, H, W)
            target_pos_neg = torch.tensor(target_pos_neg, dtype=torch.int32).to(device) # Shape (B, 1)
            
            # Flip all points  (b, y, x) -> (b, x, y) to match the format of the model
            target_points = torch.flip(target_points, dims=[2])
            
            # Normalize the points (B, X, Y) to the video resolution
            target_points = target_points / torch.tensor([video_width, video_height]).to(device)
            # Scale the (normalized) coordinates by the model's internal image size
            # target_points = target_points * predictor.image_size
            target_point_inputs = {"point_coords": target_points, "point_labels": target_labels}
            # ======================================================================================
            

            # ======================================================================================
            # FORWARD MODEL
            target_pred_masks, my_iou_pred = model(query_img, query_resized_binary_mask, target_img, target_point_inputs)
            # ======================================================================================
            
            
            # Resize predicted target mask to the original video resolution
            target_any_res_masks = target_pred_masks.to(device, non_blocking=True)
            if target_any_res_masks.shape[-2:] == (video_height, video_width):
                target_video_res_masks = target_any_res_masks
            else:
                target_video_res_masks = torch.nn.functional.interpolate(
                    target_any_res_masks,
                    size=(video_height, video_width),
                    mode="bilinear",
                    align_corners=False,
                )
            # if model.non_overlap_masks:
            #     print("WE DONT REACH THIS")
            #     target_video_res_masks = model._apply_non_overlapping_constraints(target_video_res_masks)
            
            target_video_res_masks_binary = (target_video_res_masks > 0.0).squeeze()
            
                
            # IOU PREDICTION ============================================================
            # At the moment, from multi-mask prediction, we just use the first mask for the iou prediction
            my_iou_pred = my_iou_pred[:, 0:1].squeeze()
            # Compute real IOU between the predicted mask and the closest annotation
            iou = compute_ious(pred_masks=target_video_res_masks_binary, gt_masks=target_closest_anns)
            # Abs and mean to get a scalar
            iou_loss = torch.abs(my_iou_pred - iou).mean()
            print(f"IOU loss: {iou_loss}")
            # ---------------------------------------------------------------------------------------------------------------
            # Apply back propagation
            model.zero_grad()
            scaler.scale(iou_loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
    
    
if __name__ == "__main__":
    
    args = get_args_parser()
    main(args)
