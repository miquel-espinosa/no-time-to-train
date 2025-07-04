import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from coco_dataloader import CocoDataset
from torchvision import transforms
import argparse
from sam2.utils.misc import load_video_frames
import pycocotools.mask as mask_utils
from sam2.utils.misc import fill_holes_in_mask_scores
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def get_args_parser():
    parser = argparse.ArgumentParser("Mask comparison between video and image inference")
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
    # If there are more than one channel, merge them into a single channel
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
        
        # Convert polygon segmentation to a binary mask
        rle = mask_utils.frPyObjects(annotations[0]['segmentation'], heights[0], widths[0])
        binary_mask = mask_utils.decode(rle).squeeze()
        if binary_mask.ndim == 3:
            binary_mask = binary_mask.max(axis=2)
        # Sample the center of the mask
        ylist_indices = np.where(binary_mask==1)[0]
        xlist_indices = np.where(binary_mask==1)[1]
        center_point = [xlist_indices.mean(), ylist_indices.mean()]
        if binary_mask[int(center_point[1]), int(center_point[0])] != 1:
            print(f"Center point {center_point} is not inside the mask")
            # Sample a new point, inside the mask
            new_center_point = np.random.choice(xlist_indices), np.random.choice(ylist_indices)
            print(f"New center point {new_center_point} is inside the mask")
            center_point = new_center_point
        
        input_point = np.array(center_point)
        input_label = np.array([1])
        
        # Video inference
        inference_state = predictor.init_state(video_path=None, img_paths=[f'./data/{split}/{img_info[0]["file_name"]}'])
        expanded_img, pred_backbone_out, pred_vision_features, pred_vision_pos_embeds, pred_feature_sizes = predictor._get_image_feature(inference_state, 0, args.batch_size)
        # Add new mask to inference state
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=0,
            points=[input_point],
            labels=input_label,
        )
        video_segments = {}
        for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
            video_segments[frame_idx] = {
                obj_id: (masks[j] > 0.0).squeeze().cpu().numpy()
                for j, obj_id in enumerate(object_ids)
            }
        frame_idx, masks = list(video_segments.items())[0] # Select the last frame
        print("Number of masks predicted with video inference: ", len(masks))
        out_mask_logits_binary = (out_mask_logits > 0.0).squeeze().cpu().numpy()
        mask1 = out_mask_logits_binary.squeeze()
        mask2 = masks[0].squeeze()
        print(mask1.shape, mask2.shape)
        print(mask1.sum() == mask2.sum(), mask1.sum(), mask2.sum())
        print(mask1 == mask2)
        
        # Image inference
        predictor2 = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            img = Image.open(f'./data/{split}/{img_info[0]["file_name"]}')
            print(img.size)
            predictor2.set_image(img)
            masks, scores, logits = predictor2.predict(
                point_coords=[input_point],
                point_labels=input_label,
                multimask_output=True,
            )
            
        print("Number of masks predicted with image inference: ", len(masks))
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        mask3 = masks[0].squeeze()
        print(mask2.shape, mask3.shape)
        print(mask2.sum() == mask3.sum(), mask2.sum(), mask3.sum())
        print(mask2 == mask3)
        
        # Load the original image
        img = Image.open(f'./data/{split}/{img_info[0]["file_name"]}')
        img_array = np.array(img)

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Comparison of Masks')

        # Create a colormap for the mask overlay
        colors = [(0, 0, 0, 0), (1, 0, 0, 0.5)]  # Transparent to semi-transparent red
        cmap = ListedColormap(colors)

        # Plot for mask1 (out_mask_logits_binary)
        axs[0].imshow(img_array)
        axs[0].imshow(mask1, cmap=cmap)
        axs[0].set_title('Mask 1 (Video Inference)')
        axs[0].axis('off')

        # Plot for mask2 (video propagation)
        axs[1].imshow(img_array)
        axs[1].imshow(mask2, cmap=cmap)
        axs[1].set_title('Mask 2 (Video Propagation)')
        axs[1].axis('off')

        # Plot for mask3 (image inference)
        axs[2].imshow(img_array)
        axs[2].imshow(mask3, cmap=cmap)
        axs[2].set_title('Mask 3 (Image Inference)')
        axs[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'mask_comparison_image_video/mask_comparison_{img_info[0]["file_name"]}.png')
        plt.close()

        print(f"Mask comparison plot saved as 'mask_comparison_{img_info[0]['file_name']}.png'")

        break

if __name__ == "__main__":
    args = get_args_parser()
    main(args)