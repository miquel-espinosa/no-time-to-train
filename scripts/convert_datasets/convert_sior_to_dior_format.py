#!/usr/bin/env python3
"""
Convert SIOR dataset to DIOR format for the pipeline.

SIOR format:
- JPEGImages-trainval/: Training images (00001.jpg - 11725.jpg)
- JPEGImages-test/: Test images (11726.jpg onwards)
- ins/: Contains .pkl files with instance annotations (RLE masks, bboxes, categories)
  Note: Each pkl file contains a single annotation dict (not a list like FAST)

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
            # SIOR pkl files can be a single dict or a list
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


def process_split(images_dir, ins_dir, output_img_dir, image_files, categories_dict, coco_categories):
    """Process a split (train or test) and create COCO format annotations."""
    
    print(f"\nProcessing split: {len(image_files)} images")
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    
    for img_idx, img_filename in enumerate(tqdm(image_files, desc="Processing")):
        image_id = img_idx + 1
        
        # Get image dimensions
        img_path = Path(images_dir) / img_filename
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
        # Image is like 00001.jpg, pkl is like 00001.pkl
        img_stem = Path(img_filename).stem
        pkl_filename = f"{img_stem}.pkl"
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
    
    # Create COCO format dictionary
    coco_dict = {
        'images': coco_images,
        'type': 'instances',
        'annotations': coco_annotations,
        'categories': coco_categories
    }
    
    return coco_dict


def main():
    parser = argparse.ArgumentParser(description='Convert SIOR dataset to DIOR format')
    parser.add_argument('--sior_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/SIOR',
                        help='Path to SIOR dataset (with original_files subfolder)')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: same as sior_dir)')
    args = parser.parse_args()
    
    sior_dir = Path(args.sior_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sior_dir
    
    original_files = sior_dir / 'original_files'
    train_images_dir = original_files / 'JPEGImages-trainval'
    test_images_dir = original_files / 'JPEGImages-test'
    ins_dir = original_files / 'ins'
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"SIOR dataset: {sior_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all categories
    categories_dict = get_all_categories(ins_dir)
    print(f"\nFound {len(categories_dict)} categories:")
    for cat_name, cat_id in sorted(categories_dict.items(), key=lambda x: x[1]):
        print(f"  {cat_id}: {cat_name}")
    
    # Create COCO categories list
    coco_categories = create_coco_categories(categories_dict)
    
    # Get image file lists
    train_image_files = sorted([f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png'))])
    test_image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"\nTrain images: {len(train_image_files)}")
    print(f"Test images: {len(test_image_files)}")
    
    # Process training split
    train_coco = process_split(
        train_images_dir, ins_dir, train_dir,
        train_image_files, categories_dict, coco_categories
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
        test_images_dir, ins_dir, test_dir,
        test_image_files, categories_dict, coco_categories
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
