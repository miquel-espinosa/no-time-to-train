#!/usr/bin/env python3
"""
Convert NWPU-VHR-10 dataset to DIOR format for the pipeline.

NWPU-VHR-10 format:
- positive-image-set/: Images with objects (650 images)
- negative-image-set/: Images without objects (150 images) - we skip these
- ground-truth/: TXT files with bbox annotations
  Format: (x1,y1),(x2,y2),class_id
  Classes: 1-airplane, 2-ship, 3-storage tank, 4-baseball diamond, 
           5-tennis court, 6-basketball court, 7-ground track field, 
           8-harbor, 9-bridge, 10-vehicle

Note: NWPU-VHR-10 only has bounding boxes, NO segmentation masks.
Segmentation field is left empty (not faked from bboxes).

DIOR format output:
- train/: Training images (80%)
- test/: Test images (20%)
- annotations/train.json: COCO format annotations
- annotations/test.json: COCO format annotations
"""

import os
import json
import re
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import random


def parse_annotation_file(txt_path):
    """Parse NWPU-VHR-10 annotation file."""
    annotations = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse format: (x1,y1),(x2,y2),class_id
            # Handle spaces in coordinates
            match = re.match(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*(\d+)', line)
            if match:
                x1, y1, x2, y2, class_id = map(int, match.groups())
                annotations.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, w, h]
                    'category_id': class_id,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
    
    return annotations


def main():
    parser = argparse.ArgumentParser(description='Convert NWPU-VHR-10 dataset to DIOR format')
    parser.add_argument('--nwpu_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/NWPU-VHR-10',
                        help='Path to NWPU-VHR-10 dataset')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: reorganize in place)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of images for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for train/test split')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    nwpu_dir = Path(args.nwpu_dir)
    
    # Move original files if needed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        original_files = nwpu_dir
    else:
        original_files = nwpu_dir / 'original_files'
        if not original_files.exists():
            print("Moving original files to original_files/")
            original_files.mkdir(exist_ok=True)
            
            for item in ['ground-truth', 'positive-image-set', 'negative-image-set', 'readme.txt']:
                src = nwpu_dir / item
                if src.exists():
                    shutil.move(str(src), str(original_files / item))
        
        output_dir = nwpu_dir
    
    images_dir = original_files / 'positive-image-set'
    gt_dir = original_files / 'ground-truth'
    
    # Define categories
    categories = [
        {'supercategory': 'none', 'id': 1, 'name': 'airplane'},
        {'supercategory': 'none', 'id': 2, 'name': 'ship'},
        {'supercategory': 'none', 'id': 3, 'name': 'storage-tank'},
        {'supercategory': 'none', 'id': 4, 'name': 'baseball-diamond'},
        {'supercategory': 'none', 'id': 5, 'name': 'tennis-court'},
        {'supercategory': 'none', 'id': 6, 'name': 'basketball-court'},
        {'supercategory': 'none', 'id': 7, 'name': 'ground-track-field'},
        {'supercategory': 'none', 'id': 8, 'name': 'harbor'},
        {'supercategory': 'none', 'id': 9, 'name': 'bridge'},
        {'supercategory': 'none', 'id': 10, 'name': 'vehicle'},
    ]
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"NWPU-VHR-10 dataset: {original_files}")
    print(f"Output directory: {output_dir}")
    
    # Get all annotation files
    gt_files = sorted(gt_dir.glob("*.txt"))
    print(f"\nFound {len(gt_files)} annotation files")
    
    # Split into train/test
    random.shuffle(gt_files)
    split_idx = int(len(gt_files) * args.train_ratio)
    train_files = gt_files[:split_idx]
    test_files = gt_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    
    def process_files(gt_files, output_img_dir, split_name):
        """Process a list of annotation files."""
        coco_images = []
        coco_annotations = []
        
        for img_idx, gt_path in enumerate(tqdm(gt_files, desc=f"Processing {split_name}")):
            image_id = img_idx + 1
            img_name = gt_path.stem + '.jpg'
            src_img_path = images_dir / img_name
            
            if not src_img_path.exists():
                print(f"Image not found: {src_img_path}")
                continue
            
            # Get image dimensions
            with Image.open(src_img_path) as img:
                width, height = img.size
            
            # Copy image
            dst_img_path = output_img_dir / img_name
            if not dst_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
            
            # Add image info
            coco_images.append({
                'file_name': img_name,
                'height': height,
                'width': width,
                'id': image_id
            })
            
            # Parse annotations
            annotations = parse_annotation_file(gt_path)
            
            for ann_idx, ann in enumerate(annotations):
                bbox = ann['bbox']
                area = bbox[2] * bbox[3]
                
                # No segmentation - NWPU-VHR-10 only has bboxes
                coco_annotations.append({
                    'id': len(coco_annotations) + 1,
                    'image_id': image_id,
                    'category_id': ann['category_id'],
                    'segmentation': [],  # Empty - no instance masks in original dataset
                    'area': float(area),
                    'bbox': [float(x) for x in bbox],
                    'iscrowd': 0,
                    'ignore': 0
                })
        
        return coco_images, coco_annotations
    
    # Process train
    train_images, train_annotations = process_files(train_files, train_dir, "train")
    
    # Process test
    test_images, test_annotations = process_files(test_files, test_dir, "test")
    
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
        'images': test_images,
        'type': 'instances',
        'annotations': test_annotations,
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
    print(f"  - annotations/test.json ({len(test_images)} images, {len(test_annotations)} annotations)")
    print(f"  - train/ ({len(train_images)} images)")
    print(f"  - test/ ({len(test_images)} images)")
    print("\nNote: NWPU-VHR-10 has bboxes only, no segmentation masks (segmentation field left empty).")


if __name__ == '__main__':
    main()
