#!/usr/bin/env python3
"""
Convert SOTA dataset to DIOR format for the pipeline.

SOTA format:
- images/: Contains images (PXXXX_XXXX.png)
- ins/: Contains .pkl files with instance annotations (list of dicts)
- train.txt: List of training image IDs (without extension)
- valid.txt: List of validation image IDs (without extension)

DIOR format output:
- train/: Training images
- test/: Test images  
- annotations/train.json: COCO format annotations for training
- annotations/test.json: COCO format annotations for test
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import shutil


def get_all_categories(ins_dir):
    """Scan all pkl files to get all unique categories with their labels."""
    categories = {}
    print("Scanning for all categories...")
    pkl_files = list(Path(ins_dir).glob("*.pkl"))
    
    for pkl_path in tqdm(pkl_files, desc="Scanning categories"):
        try:
            data = pickle.load(open(pkl_path, 'rb'))
            # Handle both single dict and list of dicts
            if isinstance(data, dict):
                data = [data]
            for obj in data:
                cat_name = obj['category']
                cat_label = obj['label']
                if cat_name not in categories:
                    categories[cat_name] = cat_label
        except Exception as e:
            print(f"Error reading {pkl_path}: {e}")
            continue
    
    return categories


def create_coco_categories(categories_dict):
    """Create COCO format categories list."""
    # Sort by label ID to ensure consistent ordering
    sorted_cats = sorted(categories_dict.items(), key=lambda x: x[1])
    coco_categories = []
    for cat_name, cat_id in sorted_cats:
        coco_categories.append({
            'supercategory': 'none',
            'id': cat_id + 1,  # COCO uses 1-indexed category IDs
            'name': cat_name
        })
    return coco_categories


def convert_bbox_to_coco(bbox):
    """
    Convert bbox [x1, y1, x2, y2] to COCO bbox [x, y, width, height].
    """
    x1, y1, x2, y2 = bbox
    x = float(min(x1, x2))
    y = float(min(y1, y2))
    width = float(abs(x2 - x1))
    height = float(abs(y2 - y1))
    return [x, y, width, height]


def read_split_file(split_file):
    """Read image IDs from a split file (train.txt or valid.txt)."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def process_split(images_dir, ins_dir, output_img_dir, image_ids, categories_dict, coco_categories):
    """Process a split (train or test) and create COCO format annotations."""
    
    print(f"\nProcessing split: {len(image_ids)} images")
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    missing_images = 0
    
    for img_idx, img_id in enumerate(tqdm(image_ids, desc="Processing")):
        image_id = img_idx + 1
        
        # Find the image file (try .png first, then .jpg)
        img_filename = f"{img_id}.png"
        img_path = Path(images_dir) / img_filename
        
        if not img_path.exists():
            img_filename = f"{img_id}.jpg"
            img_path = Path(images_dir) / img_filename
        
        if not img_path.exists():
            missing_images += 1
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue
        
        # Copy image to output directory
        output_img_path = Path(output_img_dir) / img_filename
        if not output_img_path.exists():
            shutil.copy2(img_path, output_img_path)
        
        # Add to COCO images
        coco_images.append({
            'file_name': img_filename,
            'height': height,
            'width': width,
            'id': image_id
        })
        
        # Get corresponding pkl annotation file
        pkl_filename = f"{img_id}.pkl"
        pkl_path = Path(ins_dir) / pkl_filename
        
        if pkl_path.exists():
            try:
                data = pickle.load(open(pkl_path, 'rb'))
                
                # Handle both single dict and list of dicts
                if isinstance(data, dict):
                    annotations = [data]
                else:
                    annotations = data
                
                for obj in annotations:
                    # Convert bbox
                    bbox = convert_bbox_to_coco(obj['bbox'])
                    area = bbox[2] * bbox[3]  # width * height
                    
                    # Get category ID (1-indexed for COCO)
                    category_id = obj['label'] + 1
                    
                    # Get segmentation mask (already in RLE format)
                    segmentation = obj['mask']
                    
                    coco_annotations.append({
                        'area': area,
                        'iscrowd': 0,
                        'image_id': image_id,
                        'bbox': bbox,
                        'category_id': category_id,
                        'id': annotation_id,
                        'ignore': 0,
                        'segmentation': segmentation  # RLE format
                    })
                    annotation_id += 1
                    
            except Exception as e:
                print(f"Error reading annotations {pkl_path}: {e}")
                continue
    
    if missing_images > 0:
        print(f"Warning: {missing_images} images not found")
    
    # Create COCO format dictionary
    coco_dict = {
        'images': coco_images,
        'type': 'instances',
        'annotations': coco_annotations,
        'categories': coco_categories
    }
    
    return coco_dict


def main():
    parser = argparse.ArgumentParser(description='Convert SOTA dataset to DIOR format')
    parser.add_argument('--sota_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/SOTA',
                        help='Path to SOTA dataset (with original_files subfolder)')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: same as sota_dir)')
    args = parser.parse_args()
    
    sota_dir = Path(args.sota_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sota_dir
    
    original_files = sota_dir / 'original_files'
    images_dir = original_files / 'images'
    ins_dir = original_files / 'ins'
    train_split_file = original_files / 'train.txt'
    valid_split_file = original_files / 'valid.txt'
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"SOTA dataset: {sota_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all categories
    categories_dict = get_all_categories(ins_dir)
    print(f"\nFound {len(categories_dict)} categories:")
    for cat_name, cat_id in sorted(categories_dict.items(), key=lambda x: x[1]):
        print(f"  {cat_id}: {cat_name}")
    
    # Create COCO categories list
    coco_categories = create_coco_categories(categories_dict)
    
    # Read split files
    train_ids = read_split_file(train_split_file)
    test_ids = read_split_file(valid_split_file)
    
    print(f"\nTrain IDs from split file: {len(train_ids)}")
    print(f"Test IDs from split file: {len(test_ids)}")
    
    # Process training split
    train_coco = process_split(
        images_dir, ins_dir, train_dir,
        train_ids, categories_dict, coco_categories
    )
    
    # Save train.json
    train_json_path = annotations_dir / 'train.json'
    print(f"\nSaving train annotations to {train_json_path}")
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f)
    print(f"  Images: {len(train_coco['images'])}")
    print(f"  Annotations: {len(train_coco['annotations'])}")
    
    # Process test split
    test_coco = process_split(
        images_dir, ins_dir, test_dir,
        test_ids, categories_dict, coco_categories
    )
    
    # Save test.json
    test_json_path = annotations_dir / 'test.json'
    print(f"\nSaving test annotations to {test_json_path}")
    with open(test_json_path, 'w') as f:
        json.dump(test_coco, f)
    print(f"  Images: {len(test_coco['images'])}")
    print(f"  Annotations: {len(test_coco['annotations'])}")
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - annotations/train.json")
    print(f"  - annotations/test.json")
    print(f"  - train/ ({len(train_coco['images'])} images)")
    print(f"  - test/ ({len(test_coco['images'])} images)")


if __name__ == '__main__':
    main()
