#!/usr/bin/env python3
"""
Tile SODAA dataset images into smaller chips.

This script takes the existing SODAA dataset (already in DIOR format with large images)
and tiles it into smaller images suitable for the pipeline.

Input:
- train/: Large training images (~4800x2744)
- test/: Large test images
- annotations/train_bbox.json: COCO format annotations
- annotations/test_bbox.json: COCO format annotations

Output:
- train_tiled/: Tiled training images
- test_tiled/: Tiled test images  
- annotations/train.json: COCO format annotations for tiles
- annotations/test.json: COCO format annotations for tiles
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np


def clip_box_to_tile(bbox_xywh, tile_x, tile_y, tile_size):
    """
    Clip a bounding box to a tile and convert to tile-relative coordinates.
    
    Args:
        bbox_xywh: [x, y, width, height] in original image coordinates
        tile_x, tile_y: Top-left corner of tile in original image
        tile_size: Size of the tile
    
    Returns:
        Clipped bbox in tile coordinates [x, y, w, h] or None if invalid
    """
    x, y, w, h = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Convert to tile-relative coordinates
    x1_rel = x1 - tile_x
    y1_rel = y1 - tile_y
    x2_rel = x2 - tile_x
    y2_rel = y2 - tile_y
    
    # Clip to tile boundaries
    x1_clip = max(0, x1_rel)
    y1_clip = max(0, y1_rel)
    x2_clip = min(tile_size, x2_rel)
    y2_clip = min(tile_size, y2_rel)
    
    # Check if box is valid (has positive area)
    if x2_clip <= x1_clip or y2_clip <= y1_clip:
        return None
    
    # Calculate how much of the original box is preserved
    original_area = w * h
    clipped_w = x2_clip - x1_clip
    clipped_h = y2_clip - y1_clip
    clipped_area = clipped_w * clipped_h
    
    # Skip if less than 30% of original box is in tile
    if original_area > 0 and clipped_area / original_area < 0.3:
        return None
    
    # Skip very small boxes (less than 4x4 pixels)
    if clipped_w < 4 or clipped_h < 4:
        return None
    
    return [float(x1_clip), float(y1_clip), float(clipped_w), float(clipped_h)]


def tile_image_and_annotations(image_path, annotations, tile_size, overlap, output_dir, 
                                image_prefix, start_image_id, start_ann_id):
    """
    Tile a single image and its annotations.
    
    Returns:
        List of (image_info, annotations) tuples, next_image_id, next_ann_id
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    stride = tile_size - overlap
    tiles = []
    
    current_image_id = start_image_id
    current_ann_id = start_ann_id
    
    # Generate tile coordinates
    y_positions = list(range(0, max(1, img_height - overlap), stride))
    x_positions = list(range(0, max(1, img_width - overlap), stride))
    
    # Ensure we cover the entire image
    if len(y_positions) > 0 and y_positions[-1] + tile_size < img_height:
        y_positions.append(max(0, img_height - tile_size))
    if len(x_positions) > 0 and x_positions[-1] + tile_size < img_width:
        x_positions.append(max(0, img_width - tile_size))
    
    for tile_y in y_positions:
        for tile_x in x_positions:
            # Ensure tile doesn't go outside image
            tile_y = max(0, min(tile_y, max(0, img_height - tile_size)))
            tile_x = max(0, min(tile_x, max(0, img_width - tile_size)))
            
            # Actual tile dimensions (may be smaller at edges)
            actual_tile_w = min(tile_size, img_width - tile_x)
            actual_tile_h = min(tile_size, img_height - tile_y)
            
            # Check which annotations fall in this tile
            tile_annotations = []
            for ann in annotations:
                clipped_box = clip_box_to_tile(ann['bbox'], tile_x, tile_y, tile_size)
                if clipped_box is not None:
                    tile_annotations.append({
                        'id': current_ann_id,
                        'image_id': current_image_id,
                        'category_id': ann['category_id'],
                        'bbox': clipped_box,
                        'area': clipped_box[2] * clipped_box[3],
                        'iscrowd': ann.get('iscrowd', 0),
                        'ignore': ann.get('ignore', 0),
                        'segmentation': []  # SODAA has no segmentation
                    })
                    current_ann_id += 1
            
            # Skip tiles with no annotations
            if not tile_annotations:
                continue
            
            # Extract tile
            tile_img = img.crop((tile_x, tile_y, tile_x + actual_tile_w, tile_y + actual_tile_h))
            
            # Pad if needed
            if actual_tile_w < tile_size or actual_tile_h < tile_size:
                padded = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile_img, (0, 0))
                tile_img = padded
            
            # Convert to RGB if needed
            if tile_img.mode != 'RGB':
                tile_img = tile_img.convert('RGB')
            
            tile_filename = f"{image_prefix}_{tile_x}_{tile_y}.jpg"
            tile_path = output_dir / tile_filename
            tile_img.save(tile_path, 'JPEG', quality=95)
            
            # Create image info
            image_info = {
                'id': current_image_id,
                'file_name': tile_filename,
                'width': tile_size,
                'height': tile_size
            }
            
            tiles.append((image_info, tile_annotations))
            current_image_id += 1
    
    img.close()
    return tiles, current_image_id, current_ann_id


def process_split(input_json, input_img_dir, output_img_dir, tile_size, overlap, split_name):
    """Process a split: tile images and update annotations."""
    
    # Load existing annotations
    print(f"\nLoading {input_json}...")
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    
    # Create image id to info mapping
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Create output directory
    output_img_dir.mkdir(parents=True, exist_ok=True)
    
    all_image_infos = []
    all_annotations = []
    current_image_id = 1
    current_ann_id = 1
    
    print(f"Processing {split_name}: {len(coco_data['images'])} images")
    
    for img_info in tqdm(coco_data['images'], desc=f"Tiling {split_name}"):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = input_img_dir / img_filename
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Get annotations for this image
        img_anns = anns_by_image.get(img_id, [])
        
        # Tile the image
        image_prefix = Path(img_filename).stem
        tiles, current_image_id, current_ann_id = tile_image_and_annotations(
            img_path, img_anns, tile_size, overlap,
            output_img_dir, image_prefix, current_image_id, current_ann_id
        )
        
        for img_info_tile, tile_anns in tiles:
            all_image_infos.append(img_info_tile)
            all_annotations.extend(tile_anns)
    
    return all_image_infos, all_annotations, coco_data['categories']


def main():
    parser = argparse.ArgumentParser(
        description='Tile SODAA dataset into smaller images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python tile_sodaa_dataset.py --tile_size 512 --overlap 64

This will:
1. Load existing SODAA annotations
2. Tile large images into 512x512 chips with 64px overlap
3. Adjust bounding boxes for each tile
4. Save new annotations as train.json and test.json
        """
    )
    parser.add_argument('--sodaa_dir', type=str,
                        default='/home/s2254242/projects/no-time-to-train/data/SODAA',
                        help='Path to SODAA dataset')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Size of image tiles (default: 512)')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between tiles in pixels (default: 64)')
    parser.add_argument('--backup_annotations', action='store_true', default=True,
                        help='Backup existing annotations before overwriting')
    args = parser.parse_args()
    
    sodaa_dir = Path(args.sodaa_dir)
    
    # Input paths
    train_img_dir = sodaa_dir / 'train'
    test_img_dir = sodaa_dir / 'test'
    annotations_dir = sodaa_dir / 'annotations'
    
    # Check for existing annotations - try different naming conventions
    train_json = annotations_dir / 'train_bbox.json'
    test_json = annotations_dir / 'test_bbox.json'
    
    if not train_json.exists():
        train_json = annotations_dir / 'train.json'
    if not test_json.exists():
        test_json = annotations_dir / 'test.json'
    
    # Check paths exist
    for p in [train_img_dir, test_img_dir, train_json, test_json]:
        if not p.exists():
            print(f"ERROR: Not found: {p}")
            return
    
    # Output paths - create new tiled directories
    train_tiled_dir = sodaa_dir / 'train_tiled'
    test_tiled_dir = sodaa_dir / 'test_tiled'
    
    print(f"SODAA dataset: {sodaa_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap}px")
    
    # Process training split
    train_images, train_annotations, categories = process_split(
        train_json, train_img_dir, train_tiled_dir,
        args.tile_size, args.overlap, "train"
    )
    
    # Process test split
    test_images, test_annotations, _ = process_split(
        test_json, test_img_dir, test_tiled_dir,
        args.tile_size, args.overlap, "test"
    )
    
    # Backup existing annotations
    if args.backup_annotations:
        for json_file in ['train.json', 'test.json']:
            src = annotations_dir / json_file
            if src.exists():
                backup = annotations_dir / f'{json_file}.backup'
                if not backup.exists():
                    import shutil
                    shutil.copy2(src, backup)
                    print(f"Backed up {src} to {backup}")
    
    # Save new annotations
    print("\n=== Saving Annotations ===")
    
    train_coco = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories,
        'type': 'instances'
    }
    with open(annotations_dir / 'train.json', 'w') as f:
        json.dump(train_coco, f)
    print(f"Saved: {annotations_dir / 'train.json'}")
    
    test_coco = {
        'images': test_images,
        'annotations': test_annotations,
        'categories': categories,
        'type': 'instances'
    }
    with open(annotations_dir / 'test.json', 'w') as f:
        json.dump(test_coco, f)
    print(f"Saved: {annotations_dir / 'test.json'}")
    
    # Rename directories: replace old train/test with tiled versions
    print("\n=== Reorganizing directories ===")
    
    # Move old directories to backup
    old_train = sodaa_dir / 'train_original'
    old_test = sodaa_dir / 'test_original'
    
    if not old_train.exists() and train_img_dir.exists():
        train_img_dir.rename(old_train)
        print(f"Renamed {train_img_dir} -> {old_train}")
    
    if not old_test.exists() and test_img_dir.exists():
        test_img_dir.rename(old_test)
        print(f"Renamed {test_img_dir} -> {old_test}")
    
    # Rename tiled to train/test
    if train_tiled_dir.exists():
        train_tiled_dir.rename(train_img_dir)
        print(f"Renamed {train_tiled_dir} -> {train_img_dir}")
    
    if test_tiled_dir.exists():
        test_tiled_dir.rename(test_img_dir)
        print(f"Renamed {test_tiled_dir} -> {test_img_dir}")
    
    # Update info.txt
    info_content = f"""Dataset: SODA-A (Small Object Detection in Aerial images)
Number of classes: 9
Class names: airplane, helicopter, small-vehicle, large-vehicle, ship, container, storage-tank, swimming-pool, windmill
Has bounding boxes: Yes
Has instance segmentation masks: No
Segmentation format: None (bbox only - original has oriented bboxes converted to axis-aligned)

Source: https://shaunyuan22.github.io/SODA/
Images from: Aerial/satellite imagery

Original images: Large (~4800x2744), tiled into {args.tile_size}x{args.tile_size} chips
Tile size: {args.tile_size}x{args.tile_size} pixels
Tile overlap: {args.overlap} pixels

Train split: {len(train_images)} tiles, {len(train_annotations)} annotations
Test split: {len(test_images)} tiles, {len(test_annotations)} annotations

Note: Images split at tile boundaries, bounding boxes clipped accordingly.
Note: Original large images backed up in train_original/ and test_original/

Citation:
@inproceedings{{cheng2023sodaa,
  title={{Towards Large-Scale Small Object Detection: Survey and Benchmarks}},
  author={{Cheng, Gong and Yuan, Xiang and Yao, Xiwen and Yan, Kebing and Zeng, Qinghua and Han, Junwei}},
  journal={{IEEE TPAMI}},
  year={{2023}}
}}
"""
    
    with open(sodaa_dir / 'info.txt', 'w') as f:
        f.write(info_content)
    print(f"Updated: {sodaa_dir / 'info.txt'}")
    
    print("\n" + "="*60)
    print("Tiling complete!")
    print(f"Output directory: {sodaa_dir}")
    print(f"  - train/: {len(train_images)} image tiles")
    print(f"  - test/: {len(test_images)} image tiles")
    print(f"  - annotations/train.json: {len(train_annotations)} annotations")
    print(f"  - annotations/test.json: {len(test_annotations)} annotations")
    print(f"  - train_original/: Original large images (backup)")
    print(f"  - test_original/: Original large images (backup)")


if __name__ == '__main__':
    main()
