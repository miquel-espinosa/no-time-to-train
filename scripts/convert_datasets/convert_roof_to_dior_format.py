#!/usr/bin/env python3
"""
Convert ROOF dataset to DIOR format for the pipeline.

ROOF format:
- train/image/: Training images (TIF, 10000x10000 RGB)
- train/label/: Training masks (TIF, 10000x10000 binary: 0=background, 1=roof)
- val/image/: Validation images
- val/label/: Validation masks
- test/image/: Test images
- test/label/: Test masks (some have _vis suffix for visualization - skip these)

DIOR format output:
- train/: Training images (converted to PNG)
- val/: Validation images (converted to PNG)
- test/: Test images (converted to PNG)
- annotations/train.json: COCO format annotations for training
- annotations/val.json: COCO format annotations for validation
- annotations/test.json: COCO format annotations for test

Note: This script extracts instance masks from binary segmentation masks using connected components.
Each connected region becomes a separate roof instance.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Use OpenCV for faster I/O
import cv2

from scipy import ndimage
from pycocotools import mask as mask_utils


def mask_to_rle(binary_mask):
    """Convert a binary mask to COCO RLE format."""
    # Ensure mask is uint8 and Fortran contiguous
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def extract_instances_from_mask(mask_array, min_area=100):
    """
    Extract individual instances from a binary segmentation mask.
    
    Args:
        mask_array: 2D numpy array where 1=roof, 0=background
        min_area: Minimum area (in pixels) for an instance to be included
    
    Returns:
        List of dicts with 'category_id', 'mask', 'bbox', 'area'
    """
    instances = []
    
    # Ensure binary mask
    binary_mask = (mask_array > 0).astype(np.uint8)
    
    # Find connected components using OpenCV (faster than scipy)
    num_labels, labeled_array, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    
    # stats columns: [left, top, width, height, area]
    # Skip label 0 (background)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        
        if area < min_area:
            continue
        
        # Get bounding box from stats
        x = stats[label_idx, cv2.CC_STAT_LEFT]
        y = stats[label_idx, cv2.CC_STAT_TOP]
        w = stats[label_idx, cv2.CC_STAT_WIDTH]
        h = stats[label_idx, cv2.CC_STAT_HEIGHT]
        bbox = [float(x), float(y), float(w), float(h)]
        
        # Create instance mask
        instance_mask = (labeled_array == label_idx)
        
        # Convert mask to RLE
        rle = mask_to_rle(instance_mask)
        
        instances.append({
            'category_id': 1,  # Single category: roof
            'mask': rle,
            'bbox': bbox,
            'area': float(area)
        })
    
    return instances


def process_single_image(args):
    """Process a single image and its label. Designed for parallel execution."""
    img_path, label_dir, output_img_dir, min_area, image_id = args
    
    img_stem = Path(img_path).stem
    img_filename = Path(img_path).name
    
    result = {
        'image_id': image_id,
        'image_info': None,
        'annotations': [],
        'error': None,
        'no_label': False,
        'no_instances': False
    }
    
    try:
        # Load image using OpenCV (faster than PIL)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            result['error'] = f"Failed to load image: {img_path}"
            return result
        
        height, width = img.shape[:2]
        
        # Save as PNG
        new_filename = img_stem + '.png'
        output_img_path = Path(output_img_dir) / new_filename
        
        if not output_img_path.exists():
            # Convert BGR to RGB for saving
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(output_img_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        # Add image info
        result['image_info'] = {
            'file_name': new_filename,
            'height': height,
            'width': width,
            'id': image_id
        }
        
        # Find corresponding label file
        label_candidates = [
            Path(label_dir) / (img_stem + '.tif'),
            Path(label_dir) / (img_stem + '.tiff'),
            Path(label_dir) / (img_stem + '.TIF'),
        ]
        
        label_path = None
        for candidate in label_candidates:
            if candidate.exists():
                label_path = candidate
                break
        
        if label_path is None:
            result['no_label'] = True
            return result
        
        # Load label mask using OpenCV
        mask_array = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if mask_array is None:
            result['no_label'] = True
            return result
        
        # Extract instances
        instances = extract_instances_from_mask(mask_array, min_area=min_area)
        
        if len(instances) == 0:
            result['no_instances'] = True
            return result
        
        result['annotations'] = instances
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def process_split_parallel(image_dir, label_dir, output_img_dir, split_name, min_area=100, num_workers=None):
    """Process a split using parallel workers."""
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted([
        Path(image_dir) / f 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    print(f"\nProcessing {split_name} split: {len(image_files)} images")
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)  # Cap at 16 workers
    
    print(f"Using {num_workers} parallel workers")
    
    # Prepare arguments for parallel processing
    task_args = [
        (str(img_path), str(label_dir), str(output_img_dir), min_area, idx + 1)
        for idx, img_path in enumerate(image_files)
    ]
    
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    skipped_no_label = 0
    skipped_no_instances = 0
    errors = 0
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, args): args for args in task_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name}"):
            result = future.result()
            
            if result['error']:
                errors += 1
                print(f"\nError: {result['error']}")
                continue
            
            if result['image_info']:
                coco_images.append(result['image_info'])
            
            if result['no_label']:
                skipped_no_label += 1
                continue
            
            if result['no_instances']:
                skipped_no_instances += 1
                continue
            
            # Add annotations with sequential IDs
            for inst in result['annotations']:
                coco_annotations.append({
                    'id': annotation_id,
                    'image_id': result['image_id'],
                    'category_id': inst['category_id'],
                    'segmentation': inst['mask'],
                    'area': inst['area'],
                    'bbox': inst['bbox'],
                    'iscrowd': 0,
                    'ignore': 0
                })
                annotation_id += 1
    
    # Sort images by ID for consistency
    coco_images.sort(key=lambda x: x['id'])
    
    if skipped_no_label > 0:
        print(f"  Warning: {skipped_no_label} images had no matching label file")
    if skipped_no_instances > 0:
        print(f"  Note: {skipped_no_instances} images had no valid instances")
    if errors > 0:
        print(f"  Errors: {errors} images failed to process")
    
    return coco_images, coco_annotations


def main():
    parser = argparse.ArgumentParser(description='Convert ROOF dataset to DIOR format')
    parser.add_argument('--roof_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/ROOF',
                        help='Path to ROOF dataset')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: reorganize in place)')
    parser.add_argument('--min_instance_area', type=int, default=100,
                        help='Minimum instance area in pixels (default: 100)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    args = parser.parse_args()
    
    roof_dir = Path(args.roof_dir)
    
    # Move original files if reorganizing in place
    if args.output_dir:
        output_dir = Path(args.output_dir)
        original_files = roof_dir
    else:
        original_files = roof_dir / 'original_files'
        if not original_files.exists():
            print("Moving original files to original_files/")
            original_files.mkdir(exist_ok=True)
            
            # Move all existing directories and files
            for item in os.listdir(roof_dir):
                if item != 'original_files':
                    src = roof_dir / item
                    dst = original_files / item
                    shutil.move(str(src), str(dst))
        
        output_dir = roof_dir
    
    # Define source directories
    train_image_dir = original_files / 'train' / 'image'
    train_label_dir = original_files / 'train' / 'label'
    val_image_dir = original_files / 'val' / 'image'
    val_label_dir = original_files / 'val' / 'label'
    test_image_dir = original_files / 'test' / 'image'
    test_label_dir = original_files / 'test' / 'label'
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_out_dir = output_dir / 'train'
    val_out_dir = output_dir / 'val'
    test_out_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"ROOF dataset: {original_files}")
    print(f"Output directory: {output_dir}")
    
    # Define single category for roof
    coco_categories = [
        {'supercategory': 'building', 'id': 1, 'name': 'roof'}
    ]
    
    # Process training split
    if train_image_dir.exists():
        print("\n=== Processing Training Split ===")
        train_images, train_annotations = process_split_parallel(
            train_image_dir, train_label_dir, train_out_dir,
            "train", min_area=args.min_instance_area, num_workers=args.num_workers
        )
        
        train_coco = {
            'images': train_images,
            'type': 'instances',
            'annotations': train_annotations,
            'categories': coco_categories
        }
        
        train_json_path = annotations_dir / 'train.json'
        print(f"\nSaving train annotations to {train_json_path}")
        with open(train_json_path, 'w') as f:
            json.dump(train_coco, f)
        print(f"  Images: {len(train_images)}")
        print(f"  Annotations: {len(train_annotations)}")
    
    # Process validation split
    if val_image_dir.exists():
        print("\n=== Processing Validation Split ===")
        val_images, val_annotations = process_split_parallel(
            val_image_dir, val_label_dir, val_out_dir,
            "val", min_area=args.min_instance_area, num_workers=args.num_workers
        )
        
        val_coco = {
            'images': val_images,
            'type': 'instances',
            'annotations': val_annotations,
            'categories': coco_categories
        }
        
        val_json_path = annotations_dir / 'val.json'
        print(f"\nSaving val annotations to {val_json_path}")
        with open(val_json_path, 'w') as f:
            json.dump(val_coco, f)
        print(f"  Images: {len(val_images)}")
        print(f"  Annotations: {len(val_annotations)}")
    
    # Process test split
    if test_image_dir.exists():
        print("\n=== Processing Test Split ===")
        test_images, test_annotations = process_split_parallel(
            test_image_dir, test_label_dir, test_out_dir,
            "test", min_area=args.min_instance_area, num_workers=args.num_workers
        )
        
        test_coco = {
            'images': test_images,
            'type': 'instances',
            'annotations': test_annotations,
            'categories': coco_categories
        }
        
        test_json_path = annotations_dir / 'test.json'
        print(f"\nSaving test annotations to {test_json_path}")
        with open(test_json_path, 'w') as f:
            json.dump(test_coco, f)
        print(f"  Images: {len(test_images)}")
        print(f"  Annotations: {len(test_annotations)}")
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - original_files/ (original TIF data)")
    print(f"  - annotations/ (COCO format JSON files)")
    print(f"  - train/, val/, test/ (PNG images)")
    print("\nNote: Images converted from TIF to PNG format.")
    print("Note: Instance masks extracted using connected components from binary masks.")


if __name__ == '__main__':
    main()
