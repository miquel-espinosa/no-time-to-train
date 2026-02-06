#!/usr/bin/env python3
"""
Convert SODA-A dataset to DIOR format for the pipeline.

SODA-A format:
- images/: Contains all images (XXXXX.jpg)
- annotations/train/: Per-image JSON files for training
- annotations/val/: Per-image JSON files for validation  
- annotations/test/: Per-image JSON files for testing

Each JSON has:
- images: {file_name, height, width, id}
- annotations: [{poly: [8 points], area, category_id, image_id, id}, ...]
- categories: [{id, name}, ...]

Note: SODA-A has oriented bounding boxes (8-point polygons), NOT instance segmentation masks.
The 'poly' field represents rotated rectangles (4 corners = 8 coordinates).
Segmentation field is left empty since these are not true instance masks.

DIOR format output:
- train/: Training images
- test/: Test images (we combine val+test as test)
- annotations/train.json: COCO format annotations
- annotations/test.json: COCO format annotations
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil


def poly_to_bbox(poly):
    """Convert 8-point polygon to COCO bbox [x, y, width, height]."""
    xs = poly[0::2]  # x coordinates
    ys = poly[1::2]  # y coordinates
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def process_split(ann_dir, images_dir, output_img_dir, categories, split_name):
    """Process a split and create COCO format annotations."""
    
    json_files = sorted(Path(ann_dir).glob("*.json"))
    print(f"\nProcessing {split_name}: {len(json_files)} annotation files")
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    coco_images = []
    coco_annotations = []
    image_id_counter = 1
    annotation_id_counter = 1
    
    for json_path in tqdm(json_files, desc=f"Processing {split_name}"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue
        
        img_info = data['images']
        file_name = img_info['file_name']
        
        # Copy image
        src_img_path = Path(images_dir) / file_name
        dst_img_path = Path(output_img_dir) / file_name
        
        if src_img_path.exists():
            if not dst_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
            
            # Get dimensions if not in annotation
            if 'width' not in img_info or 'height' not in img_info:
                with Image.open(src_img_path) as img:
                    width, height = img.size
            else:
                width = img_info['width']
                height = img_info['height']
            
            # Add to COCO images
            coco_images.append({
                'file_name': file_name,
                'height': height,
                'width': width,
                'id': image_id_counter
            })
            
            # Process annotations
            for ann in data.get('annotations', []):
                # Skip 'ignore' category (id=9)
                if ann['category_id'] == 9:
                    continue
                
                poly = ann['poly']
                bbox = poly_to_bbox(poly)
                area = ann.get('area') or (bbox[2] * bbox[3])
                
                # No segmentation - SODA-A only has oriented bboxes, not instance masks
                coco_annotations.append({
                    'id': annotation_id_counter,
                    'image_id': image_id_counter,
                    'category_id': ann['category_id'] + 1,  # COCO uses 1-indexed
                    'segmentation': [],  # Empty - no instance masks in original dataset
                    'area': float(area) if area else bbox[2] * bbox[3],
                    'bbox': bbox,
                    'iscrowd': 0,
                    'ignore': 0
                })
                annotation_id_counter += 1
            
            image_id_counter += 1
        else:
            print(f"Image not found: {src_img_path}")
    
    return coco_images, coco_annotations


def main():
    parser = argparse.ArgumentParser(description='Convert SODA-A dataset to DIOR format')
    parser.add_argument('--soda_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/SODA-A',
                        help='Path to SODA-A dataset')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: reorganize in place)')
    args = parser.parse_args()
    
    soda_dir = Path(args.soda_dir)
    
    # Move original files if needed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        original_files = soda_dir
    else:
        original_files = soda_dir / 'original_files'
        if not original_files.exists():
            print("Moving original files to original_files/")
            original_files.mkdir(exist_ok=True)
            
            for item in ['annotations', 'images', 'README.md']:
                src = soda_dir / item
                if src.exists():
                    shutil.move(str(src), str(original_files / item))
        
        output_dir = soda_dir
    
    images_dir = original_files / 'images'
    train_ann_dir = original_files / 'annotations' / 'train'
    val_ann_dir = original_files / 'annotations' / 'val'
    test_ann_dir = original_files / 'annotations' / 'test'
    
    # Define categories (excluding 'ignore')
    categories = [
        {'supercategory': 'none', 'id': 1, 'name': 'airplane'},
        {'supercategory': 'none', 'id': 2, 'name': 'helicopter'},
        {'supercategory': 'none', 'id': 3, 'name': 'small-vehicle'},
        {'supercategory': 'none', 'id': 4, 'name': 'large-vehicle'},
        {'supercategory': 'none', 'id': 5, 'name': 'ship'},
        {'supercategory': 'none', 'id': 6, 'name': 'container'},
        {'supercategory': 'none', 'id': 7, 'name': 'storage-tank'},
        {'supercategory': 'none', 'id': 8, 'name': 'swimming-pool'},
        {'supercategory': 'none', 'id': 9, 'name': 'windmill'},
    ]
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"SODA-A dataset: {original_files}")
    print(f"Output directory: {output_dir}")
    
    # Process training split
    train_images, train_annotations = process_split(
        train_ann_dir, images_dir, train_dir, categories, "train"
    )
    
    # Process val and test together as test
    val_images, val_annotations = process_split(
        val_ann_dir, images_dir, test_dir, categories, "val"
    )
    
    test_images, test_annotations = process_split(
        test_ann_dir, images_dir, test_dir, categories, "test"
    )
    
    # Combine val and test, adjusting IDs
    combined_test_images = val_images.copy()
    combined_test_annotations = val_annotations.copy()
    
    # Adjust IDs for test set to continue from val
    max_img_id = max(img['id'] for img in val_images) if val_images else 0
    max_ann_id = max(ann['id'] for ann in val_annotations) if val_annotations else 0
    
    for img in test_images:
        img['id'] += max_img_id
        combined_test_images.append(img)
    
    for ann in test_annotations:
        ann['id'] += max_ann_id
        ann['image_id'] += max_img_id
        combined_test_annotations.append(ann)
    
    # Save train.json
    train_coco = {
        'images': train_images,
        'type': 'instances',
        'annotations': train_annotations,
        'categories': categories
    }
    
    train_json_path = annotations_dir / 'train.json'
    print(f"\nSaving train annotations to {train_json_path}")
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f)
    
    # Save test.json
    test_coco = {
        'images': combined_test_images,
        'type': 'instances',
        'annotations': combined_test_annotations,
        'categories': categories
    }
    
    test_json_path = annotations_dir / 'test.json'
    print(f"Saving test annotations to {test_json_path}")
    with open(test_json_path, 'w') as f:
        json.dump(test_coco, f)
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - annotations/train.json ({len(train_images)} images, {len(train_annotations)} annotations)")
    print(f"  - annotations/test.json ({len(combined_test_images)} images, {len(combined_test_annotations)} annotations)")
    print(f"  - train/ ({len(train_images)} images)")
    print(f"  - test/ ({len(combined_test_images)} images)")
    print("\nNote: SODA-A has oriented bboxes only, no segmentation masks (segmentation field left empty).")


if __name__ == '__main__':
    main()
