#!/usr/bin/env python3
"""
Convert HRSID_JPG dataset to DIOR format for the pipeline.

HRSID is already in COCO format with instance segmentation.
Just needs reorganization:
- annotations/train2017.json -> annotations/train.json
- annotations/test2017.json -> annotations/test.json  
- JPEGImages/ -> train/ and test/

DIOR format output:
- train/: Training images
- test/: Test images  
- annotations/train.json: COCO format annotations for training
- annotations/test.json: COCO format annotations for test
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def process_split(src_json, src_images_dir, output_img_dir, output_json_path):
    """Process a split and copy images."""
    
    # Load source JSON
    with open(src_json, 'r') as f:
        data = json.load(f)
    
    print(f"\nProcessing: {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Copy images
    for img_info in tqdm(data['images'], desc="Copying images"):
        src_path = Path(src_images_dir) / img_info['file_name']
        dst_path = Path(output_img_dir) / img_info['file_name']
        
        if src_path.exists() and not dst_path.exists():
            shutil.copy2(src_path, dst_path)
    
    # The JSON is already in COCO format, just need to ensure proper structure
    # Add 'type' if missing
    if 'type' not in data:
        data['type'] = 'instances'
    
    # Ensure annotations have all required fields
    for ann in data['annotations']:
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0
        if 'ignore' not in ann:
            ann['ignore'] = 0
    
    # Save output JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(data, f)
    
    return len(data['images']), len(data['annotations'])


def main():
    parser = argparse.ArgumentParser(description='Convert HRSID_JPG dataset to DIOR format')
    parser.add_argument('--hrsid_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/HRSID_JPG',
                        help='Path to HRSID_JPG dataset')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: creates DIOR structure in same location)')
    args = parser.parse_args()
    
    hrsid_dir = Path(args.hrsid_dir)
    
    # Check if we need to reorganize in place or to a new location
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Reorganize in place - first move original files
        original_files = hrsid_dir / 'original_files'
        if not original_files.exists():
            print("Moving original files to original_files/")
            original_files.mkdir(exist_ok=True)
            
            # Move existing folders to original_files
            for item in ['annotations', 'JPEGImages', 'inshore_offshore']:
                src = hrsid_dir / item
                if src.exists():
                    shutil.move(str(src), str(original_files / item))
        
        output_dir = hrsid_dir
        hrsid_dir = original_files
    
    src_images_dir = hrsid_dir / 'JPEGImages'
    src_train_json = hrsid_dir / 'annotations' / 'train2017.json'
    src_test_json = hrsid_dir / 'annotations' / 'test2017.json'
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"HRSID dataset: {hrsid_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process training split
    print("\n=== Processing Training Split ===")
    train_imgs, train_anns = process_split(
        src_train_json, src_images_dir, train_dir,
        annotations_dir / 'train.json'
    )
    
    # Process test split
    print("\n=== Processing Test Split ===")
    test_imgs, test_anns = process_split(
        src_test_json, src_images_dir, test_dir,
        annotations_dir / 'test.json'
    )
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - annotations/train.json ({train_imgs} images, {train_anns} annotations)")
    print(f"  - annotations/test.json ({test_imgs} images, {test_anns} annotations)")
    print(f"  - train/ ({train_imgs} images)")
    print(f"  - test/ ({test_imgs} images)")


if __name__ == '__main__':
    main()
