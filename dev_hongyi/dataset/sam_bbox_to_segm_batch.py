import argparse
import json
import os
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from pycocotools import mask as mask_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Generate segmentation masks from bounding boxes using SAM')
    parser.add_argument('--input_json', type=str, required=True, help='Path to input COCO json file')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to directory containing images')
    parser.add_argument('--sam_checkpoint', type=str, required=True, help='Path to SAM checkpoint')
    parser.add_argument('--model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM model type')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run inference on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--visualize', action='store_true', help='Visualize and save results')
    return parser.parse_args()

def load_sam_model(checkpoint_path, model_type, device):
    """Load SAM model"""
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam

def prepare_batched_inputs(images, boxes, sam):
    """Prepare batched inputs for SAM model"""
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    batched_inputs = []
    
    for idx, (image, image_boxes) in enumerate(zip(images, boxes)):
        
        if len(image_boxes) == 0:
            raise ValueError(f"ERROR: No boxes found for image {idx} while attempting to segment with SAM!")
        
        # Convert COCO format [x,y,w,h] to SAM format [x1,y1,x2,y2]
        sam_boxes = []
        for box in image_boxes:
            x, y, w, h = box
            sam_boxes.append([x, y, x + w, y + h])
        sam_boxes = torch.tensor(sam_boxes, device=sam.device)
        
        transformed_image = resize_transform.apply_image(image)
        transformed_boxes = resize_transform.apply_boxes_torch(sam_boxes, image.shape[:2])
        
        batched_inputs.append({
            'image': torch.as_tensor(transformed_image, device=sam.device).permute(2, 0, 1).contiguous(),
            'boxes': transformed_boxes,
            'original_size': image.shape[:2]
        })
    
    return batched_inputs

def mask_to_rle_coco(mask):
    """Convert binary mask to COCO RLE format"""
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    return rle

def mask_to_polygon(ann, mask):
    """Convert binary mask to COCO polygon format"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:  # Valid polygons must have at least 4 values (4 is just a line)
            segmentation.append(contour)
        else:
            print(f"Warning: Invalid polygon when generating segmentation mask - Skipping")

    if not segmentation:
        # This usually happens when the bounding box is too small
        # We will use the bounding box to generate the segmentation mask
        print("\033[93mWarning: Generating segmentation mask from bounding box\033[0m")
        x, y, w, h = ann['bbox']
        # Create a rectangular mask from the bounding box
        mask = np.zeros_like(mask)
        mask[int(y):int(y+h), int(x):int(x+w)] = 1
        segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]] # Convert to polygon format

    return segmentation

def show_mask(mask, ax, random_color=False):
    """Show the mask on the given axis according to SAM's official visualization"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    mask_image = mask.reshape(mask.shape[-2:] + (1,)) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    """Show the box on the given axis according to SAM's official visualization"""
    # Convert from COCO format [x,y,w,h] to coordinates format [x1,y1,x2,y2]
    x0, y0, w, h = box
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_polygon(polygon, ax):
    """Show the polygon contour on the given axis"""
    # Reshape polygon points into x,y coordinates
    points = np.array(polygon).reshape(-1, 2)
    # Plot the polygon contour
    ax.plot(points[:, 0], points[:, 1], '-r', linewidth=2, label='Polygon Contour')

def visualize_results(image, masks, boxes, save_path, polygons=None):
    """Visualize and save results using SAM's official visualization style"""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Show image
    ax.imshow(image)
    
    # Show all masks
    for mask in masks:
        show_mask(mask, ax, random_color=True)
    
    # Show all boxes
    for box in boxes:
        show_box(box, ax)
    
    # Show all polygon contours if provided
    if polygons is not None:
        for polygon_group in polygons:
            for polygon in polygon_group:
                show_polygon(polygon, ax)
    
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_coco_annotations(args):
    # Load COCO annotations
    with open(args.input_json, 'r') as f:
        coco_data = json.load(f)
    
    # Load SAM model
    sam = load_sam_model(args.sam_checkpoint, args.model_type, args.device)
    
    # Create output directory for visualizations
    if args.visualize:
        output_dir = os.path.splitext(args.input_json)[0]
        os.makedirs(output_dir, exist_ok=True)
    
    # Group annotations by image
    image_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_anns:
            image_to_anns[image_id] = []
        image_to_anns[image_id].append(ann)
        
    # Images without annotations
    image_ids_without_anns = []
    images_without_anns = []
    for image in coco_data['images']:
        if image['id'] not in image_to_anns:
            image_ids_without_anns.append(image['id'])
            images_without_anns.append(image)

    print(f"\033[93mNumber of images without annotations: {len(images_without_anns)}\033[0m")
    coco_data['images'] = [image for image in coco_data['images'] if image['id'] not in image_ids_without_anns]
    
    # Process images in batches
    for i in tqdm(range(0, len(coco_data['images']), args.batch_size)):
        batch_images = coco_data['images'][i:i + args.batch_size]
        images = []
        boxes_batch = []
        valid_indices = []
        
        # Prepare batch data
        for img_idx, img_info in enumerate(batch_images):
            # Load image
            image_path = os.path.join(args.image_dir, img_info['file_name'])
            image = cv2.imread(image_path)
            
            if image is None:
                raise ValueError(f"Could not load image {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            
            # Get boxes for this image
            anns = image_to_anns.get(img_info['id'], [])
            boxes = [ann['bbox'] for ann in anns]
            boxes_batch.append(boxes)
            valid_indices.append(img_idx)
        
        if not images:  # Skip if no valid images in batch
            raise ValueError(f"No valid images found in batch {i} to {i + args.batch_size}")
            
        # Run SAM on batch
        batched_inputs = prepare_batched_inputs(images, boxes_batch, sam)
        batched_outputs = sam(batched_inputs, multimask_output=False)
        
        # Process outputs
        for img_idx, (img_info, image_boxes, output) in enumerate(zip([batch_images[i] for i in valid_indices], 
                                                                     boxes_batch, 
                                                                     batched_outputs)):
            masks = output['masks'].cpu().numpy()
            
            # Update annotations with segmentation
            anns = image_to_anns.get(img_info['id'], [])
            for ann_idx, (ann, mask) in enumerate(zip(anns, masks)):
                # Convert mask to COCO polygon format
                segmentation = mask_to_polygon(ann, mask[0])
                if len(segmentation) > 0:  # Only update if valid segmentation is found
                    ann['segmentation'] = segmentation
                    ann['area'] = float(mask[0].sum())
                else:
                    raise ValueError(f"ERROR: Failed to generate valid segmentation for image {img_info['file_name']}, annotation {ann_idx}")
                    # Use RLE format as fallback when polygon conversion fails
                    # rle = mask_to_rle_coco(mask[0])
                    # ann['segmentation'] = rle
                    # ann['area'] = float(mask[0].sum())
                    # ann['segmentation']['counts'] = ann['segmentation']['counts'].decode('utf-8')
                    # print(f"    ==> Using RLE format for segmentation for image {img_info['file_name']}, annotation {ann_idx}")
            
            
            # Visualize if requested
            if args.visualize:
                save_path = os.path.join(output_dir, f"{os.path.splitext(img_info['file_name'])[0]}_segm.png")
                # Get polygons for visualization
                polygons = [ann['segmentation'] for ann in anns]
                visualize_results(images[img_idx], masks[:, 0], image_boxes, save_path, polygons=polygons)
    
    # Save updated COCO annotations
    output_json = os.path.splitext(args.input_json)[0] + '_with_segm.json'
    # Add images without annotations to the COCO data
    coco_data['images'] = coco_data['images'] + images_without_anns
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)

if __name__ == '__main__':
    args = parse_args()
    process_coco_annotations(args)
