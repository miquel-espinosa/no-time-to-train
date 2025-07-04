import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch.utils.data as data
from pycocotools.coco import COCO
import argparse
from sam2.utils.misc import load_video_frames
import pycocotools.mask as mask_utils
from sam2.utils.misc import fill_holes_in_mask_scores
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json
from tqdm import tqdm
import scipy.ndimage as ndimage

class SimpleCocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, image_size):
        self.root = root
        self.image_size = image_size
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        
    def __getitem__(self, index):
        ann_id = self.ids[index]
        annotation = self.coco.anns[ann_id]
        img_id = annotation['image_id']
        img_info = self.coco.loadImgs(img_id)[0]
        height = img_info['height']
        width = img_info['width']
        return annotation, height, width, img_info

    def __len__(self):
        return len(self.ids)
    

def get_args_parser():
    parser = argparse.ArgumentParser("Mask comparison between video and image inference")
    parser.add_argument("--model", type=str, required=True, help="Model size in ['tiny', 'small', 'base_plus', 'large']")
    parser.add_argument("--plot", action="store_true", help="Plot the results")
    parser.add_argument("--iou_threshold", type=float, default=0.6, help="IOU threshold for filtering")
    parser.add_argument("--split", type=str, default='train2017', help="Split to filter")
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


def compute_iou(predicted_mask, ground_truth_mask):
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    return intersection / union if union > 0 else 0


def main(args):

    old_json_path = f'./data/annotations/instances_{args.split}.json'
    new_json_path = f'./data/annotations/instances_{args.split}_{args.model}_filtered_by_{args.iou_threshold}.json'

    # Load the SAM2 Hiera model
    model_cfg = get_model_cfg(args.model)
    checkpoint = f"./checkpoints/sam2_hiera_{args.model}.pt"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    # Load the original COCO JSON
    with open(old_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Dataloader
    train_dataset = SimpleCocoDataset(root=f'./data/{args.split}',
                                json=old_json_path,
                                image_size=predictor.image_size)
    
    # Create a dictionary to map annotation IDs to their indices in the list
    ann_id_to_index = {ann['id']: idx for idx, ann in enumerate(coco_data['annotations'])}

    # Process annotations
    for annotation, height, width, img_info in tqdm(train_dataset, desc="Processing annotations"):
        
        # Discard crowd annotations and tiny annotations
        if annotation['iscrowd'] == 1 or annotation['area'] < 100:
            ann_index = ann_id_to_index[annotation['id']]
            coco_data['annotations'][ann_index]['isimpossible'] = '1'
            continue
        
        # ================ PREPARE INPUT (point in center of mask) ================
        # Convert polygon segmentation to a binary mask
        binary_mask = convert_polygon_to_mask(annotation['segmentation'], width, height)
        # Sample the center of the mask
        ylist_indices = np.where(binary_mask==1)[0]
        xlist_indices = np.where(binary_mask==1)[1]
        center_point = [xlist_indices.mean(), ylist_indices.mean()]
        if binary_mask[int(center_point[1]), int(center_point[0])] != 1:
            # Sample a new point, inside the mask, and not close to the border
            print(f"Center point {center_point} is not inside the mask")
            # Calculate distance from border
            distance_from_border = ndimage.distance_transform_edt(binary_mask)
            # Find points not close to the border (e.g., at least 5 pixels away)
            valid_points = np.where((binary_mask == 1) & (distance_from_border >= 5))
            # Randomly choose a new center point from valid points
            if len(valid_points[0]) > 0:
                idx = np.random.choice(len(valid_points[0]))
                new_center_point = valid_points[1][idx], valid_points[0][idx]
            else:
                print(f"No points far from the border. Randomly choosing a point")
                new_center_point = np.random.choice(xlist_indices), np.random.choice(ylist_indices)
            print(f"New center point {new_center_point} is inside the mask")
            # old_center_point = center_point
            center_point = new_center_point
            
            # Plot the mask and the center point (in red) and the new center point (in green). DEBUG ONLY
            # print(annotation)
            # fig, axs = plt.subplots(1, 2, figsize=(10, 10))
            # axs[0].imshow(binary_mask)
            # axs[0].plot(old_center_point[0], old_center_point[1], 'r*', markersize=7)
            # axs[1].imshow(binary_mask)
            # axs[1].plot(new_center_point[0], new_center_point[1], 'g*', markersize=7)
            # plt.savefig(f'test_{img_info["file_name"]}_center_point.png')
            # plt.close()
            # exit()
        
        input_point = np.array([center_point])
        input_label = np.array([1])
        
        # ================ VIDEO INFERENCE ================
        inference_state = predictor.init_state(video_path=None, img_paths=[f'./data/{args.split}/{img_info["file_name"]}'])
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=input_point,
            labels=input_label,
        )
        out_mask_logits_binary = (out_mask_logits > 0.0).squeeze().cpu().numpy()
        mask = out_mask_logits_binary.squeeze()
        
        # ================ COMPUTE IOU ================
        iou = compute_iou(predicted_mask=mask, ground_truth_mask=binary_mask)
        
        # ================ UPDATE ANNOTATION ================
        ann_index = ann_id_to_index[annotation['id']]
        coco_data['annotations'][ann_index]['isimpossible'] = str(int(iou < args.iou_threshold))
        
        # ================ PLOT RESULTS ================
        if args.plot:
            # Load the original image
            img = Image.open(f'./data/{args.split}/{img_info["file_name"]}')
            img_array = np.array(img)

            # Create a figure with 3 subplots
            fig, axs = plt.subplots(1, 2, figsize=(10, 10))

            # Create a colormap for the mask overlay
            colors = [(0, 0, 0, 0), (1, 0, 0, 0.5)]  # Transparent to semi-transparent red
            cmap = ListedColormap(colors)

            # Plot for mask1 (out_mask_logits_binary)
            axs[0].imshow(img_array)
            axs[0].imshow(binary_mask, cmap=cmap)
            axs[0].set_title('Mask (Ground Truth)')
            axs[0].axis('off')
            
            # Plot the input point as a star, border white
            axs[1].imshow(img_array)
            axs[1].imshow(mask, cmap=cmap)
            axs[1].plot(center_point[0], center_point[1], 'b*', markersize=7, markeredgecolor='w')
            axs[1].set_title('Mask (Video Inference)')
            axs[1].axis('off')
            
            plt.savefig(f'test_{img_info["file_name"]}_iou_{iou:.2f}.png')
            plt.close()
            
            exit()
        
        

    # Save the updated COCO JSON
    with open(new_json_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"Updated COCO JSON saved to {new_json_path}")

if __name__ == "__main__":
    args = get_args_parser()
    main(args)