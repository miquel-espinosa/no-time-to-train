#!/usr/bin/env python3
"""
Tile ISAID dataset images into smaller chips.

ISAID images range from 800 to 4000 pixels, which can cause OOM errors.
This script tiles the dataset into smaller images suitable for the pipeline.

Handles polygon segmentation masks by:
1. Clipping polygon vertices to tile boundaries
2. Converting to tile-relative coordinates
3. Filtering out polygons that become too small

Input (after conversion to DIOR format):
- train/: Training images (800-4000 px)
- test/: Test images
- annotations/train.json: COCO format with polygon segmentation
- annotations/test.json

Output:
- train/: Tiled training images
- test/: Tiled test images
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

# Increase PIL limit for large images
Image.MAX_IMAGE_PIXELS = None


def clip_polygon_to_tile(polygon, tile_x, tile_y, tile_size):
    """
    Clip a polygon to a tile and convert to tile-relative coordinates.
    
    Uses Sutherland-Hodgman algorithm for polygon clipping.
    
    Args:
        polygon: List of [x1, y1, x2, y2, ...] coordinates
        tile_x, tile_y: Top-left corner of tile in original image
        tile_size: Size of the tile
    
    Returns:
        Clipped polygon in tile coordinates or None if invalid
    """
    if len(polygon) < 6:  # Need at least 3 points
        return None
    
    # Convert flat list to list of (x, y) tuples
    points = [(polygon[i] - tile_x, polygon[i+1] - tile_y) 
              for i in range(0, len(polygon), 2)]
    
    # Clip against each edge of the tile
    def clip_edge(points, edge_func, inside_func):
        """Clip polygon against one edge."""
        if not points:
            return []
        
        output = []
        for i in range(len(points)):
            current = points[i]
            next_pt = points[(i + 1) % len(points)]
            
            current_inside = inside_func(current)
            next_inside = inside_func(next_pt)
            
            if current_inside:
                output.append(current)
                if not next_inside:
                    output.append(edge_func(current, next_pt))
            elif next_inside:
                output.append(edge_func(current, next_pt))
        
        return output
    
    def intersect_left(p1, p2):
        if p2[0] == p1[0]:
            return (0, p1[1])
        t = (0 - p1[0]) / (p2[0] - p1[0])
        return (0, p1[1] + t * (p2[1] - p1[1]))
    
    def intersect_right(p1, p2):
        if p2[0] == p1[0]:
            return (tile_size, p1[1])
        t = (tile_size - p1[0]) / (p2[0] - p1[0])
        return (tile_size, p1[1] + t * (p2[1] - p1[1]))
    
    def intersect_top(p1, p2):
        if p2[1] == p1[1]:
            return (p1[0], 0)
        t = (0 - p1[1]) / (p2[1] - p1[1])
        return (p1[0] + t * (p2[0] - p1[0]), 0)
    
    def intersect_bottom(p1, p2):
        if p2[1] == p1[1]:
            return (p1[0], tile_size)
        t = (tile_size - p1[1]) / (p2[1] - p1[1])
        return (p1[0] + t * (p2[0] - p1[0]), tile_size)
    
    # Clip against all four edges
    points = clip_edge(points, intersect_left, lambda p: p[0] >= 0)
    points = clip_edge(points, intersect_right, lambda p: p[0] <= tile_size)
    points = clip_edge(points, intersect_top, lambda p: p[1] >= 0)
    points = clip_edge(points, intersect_bottom, lambda p: p[1] <= tile_size)
    
    if len(points) < 3:
        return None
    
    # Convert back to flat list
    result = []
    for x, y in points:
        result.extend([float(x), float(y)])
    
    return result


def polygon_area(polygon):
    """Calculate area of polygon using shoelace formula."""
    if len(polygon) < 6:
        return 0
    
    n = len(polygon) // 2
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[2*i] * polygon[2*j + 1]
        area -= polygon[2*j] * polygon[2*i + 1]
    
    return abs(area) / 2


def polygon_bbox(polygon):
    """Get bounding box of polygon as [x, y, w, h]."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


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
                                image_prefix, start_image_id, start_ann_id, min_polygon_area=50):
    """
    Tile a single image and its annotations.
    
    Args:
        image_path: Path to input image
        annotations: List of annotations for this image
        tile_size: Size of tiles
        overlap: Overlap between tiles
        output_dir: Directory to save tiles
        image_prefix: Prefix for tile filenames
        start_image_id: Starting ID for images
        start_ann_id: Starting ID for annotations
        min_polygon_area: Minimum polygon area to keep (in pixels^2)
    
    Returns:
        List of (image_info, annotations) tuples, next_image_id, next_ann_id
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # For small images, don't tile - just copy with adjusted coordinates
    if img_width <= tile_size and img_height <= tile_size:
        # Image is small enough, just save it directly
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        tile_filename = f"{image_prefix}.jpg"
        tile_path = output_dir / tile_filename
        img.save(tile_path, 'JPEG', quality=95)
        
        image_info = {
            'id': start_image_id,
            'file_name': tile_filename,
            'width': img_width,
            'height': img_height
        }
        
        tile_annotations = []
        current_ann_id = start_ann_id
        for ann in annotations:
            new_ann = {
                'id': current_ann_id,
                'image_id': start_image_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': ann.get('iscrowd', 0),
                'ignore': ann.get('ignore', 0),
                'segmentation': ann.get('segmentation', [])
            }
            tile_annotations.append(new_ann)
            current_ann_id += 1
        
        img.close()
        return [(image_info, tile_annotations)], start_image_id + 1, current_ann_id
    
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
                # First check bbox
                clipped_box = clip_box_to_tile(ann['bbox'], tile_x, tile_y, tile_size)
                if clipped_box is None:
                    continue
                
                # Process segmentation polygons
                clipped_segmentation = []
                if 'segmentation' in ann and ann['segmentation']:
                    for polygon in ann['segmentation']:
                        if isinstance(polygon, list) and len(polygon) >= 6:
                            clipped_poly = clip_polygon_to_tile(polygon, tile_x, tile_y, tile_size)
                            if clipped_poly and len(clipped_poly) >= 6:
                                area = polygon_area(clipped_poly)
                                if area >= min_polygon_area:
                                    clipped_segmentation.append(clipped_poly)
                
                # Skip if no valid segmentation and original had segmentation
                if ann.get('segmentation') and not clipped_segmentation:
                    continue
                
                # Calculate area from segmentation if available
                if clipped_segmentation:
                    area = sum(polygon_area(p) for p in clipped_segmentation)
                    # Also update bbox from segmentation
                    all_xs = []
                    all_ys = []
                    for poly in clipped_segmentation:
                        all_xs.extend(poly[0::2])
                        all_ys.extend(poly[1::2])
                    if all_xs and all_ys:
                        clipped_box = [
                            min(all_xs), min(all_ys),
                            max(all_xs) - min(all_xs),
                            max(all_ys) - min(all_ys)
                        ]
                else:
                    area = clipped_box[2] * clipped_box[3]
                
                tile_annotations.append({
                    'id': current_ann_id,
                    'image_id': current_image_id,
                    'category_id': ann['category_id'],
                    'bbox': clipped_box,
                    'area': area,
                    'iscrowd': ann.get('iscrowd', 0),
                    'ignore': ann.get('ignore', 0),
                    'segmentation': clipped_segmentation
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


def process_split(input_json, input_img_dir, output_img_dir, tile_size, overlap, 
                  split_name, min_polygon_area=50):
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
    
    # Track statistics
    images_tiled = 0
    images_copied = 0
    
    print(f"Processing {split_name}: {len(coco_data['images'])} images")
    
    for img_info in tqdm(coco_data['images'], desc=f"Tiling {split_name}"):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_path = input_img_dir / img_filename
        
        if not img_path.exists():
            # Try different extensions
            for ext in ['.png', '.jpg', '.jpeg', '.tif']:
                alt_path = img_path.with_suffix(ext)
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Get annotations for this image
        img_anns = anns_by_image.get(img_id, [])
        
        # Check if tiling is needed
        with Image.open(img_path) as check_img:
            needs_tiling = check_img.size[0] > tile_size or check_img.size[1] > tile_size
        
        if needs_tiling:
            images_tiled += 1
        else:
            images_copied += 1
        
        # Tile the image
        image_prefix = Path(img_filename).stem
        tiles, current_image_id, current_ann_id = tile_image_and_annotations(
            img_path, img_anns, tile_size, overlap,
            output_img_dir, image_prefix, current_image_id, current_ann_id,
            min_polygon_area
        )
        
        for img_info_tile, tile_anns in tiles:
            all_image_infos.append(img_info_tile)
            all_annotations.extend(tile_anns)
    
    print(f"  Images tiled: {images_tiled}, copied directly: {images_copied}")
    
    return all_image_infos, all_annotations, coco_data['categories']


def main():
    parser = argparse.ArgumentParser(
        description='Tile ISAID dataset into smaller images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python tile_isaid_dataset.py --tile_size 512 --overlap 64

This will:
1. Load existing ISAID annotations (after conversion to DIOR format)
2. Tile large images (>512px) into 512x512 chips with 64px overlap
3. Clip bounding boxes and polygon segmentation for each tile
4. Save new annotations as train.json and test.json

Note: Run convert_isaid_to_dior_format.py first to convert ISAID to DIOR format.
        """
    )
    parser.add_argument('--isaid_dir', type=str,
                        default='/home/s2254242/projects/no-time-to-train/data/ISAID',
                        help='Path to ISAID dataset (after DIOR conversion)')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Size of image tiles (default: 512)')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between tiles in pixels (default: 64)')
    parser.add_argument('--min_polygon_area', type=int, default=50,
                        help='Minimum polygon area in pixels^2 to keep (default: 50)')
    parser.add_argument('--backup_annotations', action='store_true', default=True,
                        help='Backup existing annotations before overwriting')
    args = parser.parse_args()
    
    isaid_dir = Path(args.isaid_dir)
    
    # Input paths
    train_img_dir = isaid_dir / 'train'
    test_img_dir = isaid_dir / 'test'
    annotations_dir = isaid_dir / 'annotations'
    
    train_json = annotations_dir / 'train.json'
    test_json = annotations_dir / 'test.json'
    
    # Check paths exist
    missing = []
    for p in [train_img_dir, train_json]:
        if not p.exists():
            missing.append(str(p))
    
    if missing:
        print("ERROR: The following paths were not found:")
        for p in missing:
            print(f"  {p}")
        print("\nMake sure you've run convert_isaid_to_dior_format.py first.")
        return
    
    # Output paths - create new tiled directories
    train_tiled_dir = isaid_dir / 'train_tiled'
    test_tiled_dir = isaid_dir / 'test_tiled'
    
    print(f"ISAID dataset: {isaid_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap}px")
    print(f"Min polygon area: {args.min_polygon_area}px²")
    
    # Verify that train and test have the same categories before processing
    if test_json.exists():
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        
        if train_data['categories'] != test_data['categories']:
            print("\nERROR: Train and test splits have different category mappings!")
            print("This will cause annotation labels to be mismatched.")
            print("\nTrain categories:")
            for cat in sorted(train_data['categories'], key=lambda x: x['id']):
                print(f"  id={cat['id']}: {cat['name']}")
            print("\nTest categories:")
            for cat in sorted(test_data['categories'], key=lambda x: x['id']):
                print(f"  id={cat['id']}: {cat['name']}")
            print("\nPlease re-run convert_isaid_to_dior_format.py to fix category mapping.")
            return
        else:
            print("\n✓ Train and test categories match")
    
    # Process training split
    train_images, train_annotations, categories = process_split(
        train_json, train_img_dir, train_tiled_dir,
        args.tile_size, args.overlap, "train", args.min_polygon_area
    )
    
    # Process test split if it exists
    test_images = []
    test_annotations = []
    if test_json.exists() and test_img_dir.exists():
        test_images, test_annotations, _ = process_split(
            test_json, test_img_dir, test_tiled_dir,
            args.tile_size, args.overlap, "test", args.min_polygon_area
        )
    else:
        print("\nNote: Test split not found, skipping.")
    
    # Backup existing annotations
    if args.backup_annotations:
        import shutil
        for json_file in ['train.json', 'test.json']:
            src = annotations_dir / json_file
            if src.exists():
                backup = annotations_dir / f'{json_file}.backup'
                if not backup.exists():
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
    
    if test_images:
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
    old_train = isaid_dir / 'train_original'
    old_test = isaid_dir / 'test_original'
    
    if not old_train.exists() and train_img_dir.exists():
        train_img_dir.rename(old_train)
        print(f"Renamed {train_img_dir} -> {old_train}")
    
    if test_img_dir.exists() and not old_test.exists():
        test_img_dir.rename(old_test)
        print(f"Renamed {test_img_dir} -> {old_test}")
    
    # Rename tiled to train/test
    if train_tiled_dir.exists():
        train_tiled_dir.rename(train_img_dir)
        print(f"Renamed {train_tiled_dir} -> {train_img_dir}")
    
    if test_tiled_dir.exists():
        test_tiled_dir.rename(test_img_dir)
        print(f"Renamed {test_tiled_dir} -> {test_img_dir}")
    
    # Create/update info.txt
    info_content = f"""Dataset: ISAID (Instance Segmentation in Aerial Images Dataset)
Number of classes: 15
Class names: storage_tank, Large_Vehicle, Small_Vehicle, plane, ship, Swimming_pool, 
             Harbor, tennis_court, Ground_Track_Field, Soccer_ball_field, 
             baseball_diamond, Bridge, basketball_court, Roundabout, Helicopter
Has bounding boxes: Yes
Has instance segmentation masks: Yes
Segmentation format: Polygon (clipped at tile boundaries)

Source: https://captain-whu.github.io/ISAID/
Images from: DOTA-v1.0 dataset (https://captain-whu.github.io/DOTA/)

Original images: 800-4000 pixels, tiled into {args.tile_size}x{args.tile_size} chips
Tile size: {args.tile_size}x{args.tile_size} pixels
Tile overlap: {args.overlap} pixels
Min polygon area: {args.min_polygon_area} pixels²

Train split: {len(train_images)} tiles, {len(train_annotations)} annotations
Test split: {len(test_images)} tiles, {len(test_annotations)} annotations

Note: Images smaller than {args.tile_size}x{args.tile_size} are copied directly without tiling.
Note: Original large images backed up in train_original/ and test_original/
Note: Polygon segmentation masks are clipped at tile boundaries.

Citation:
@inproceedings{{waqas2019isaid,
  title={{ISAID: A Large-scale Dataset for Instance Segmentation in Aerial Images}},
  author={{Waqas Zamir, Syed and Arora, Aditya and Gupta, Akshita and Khan, Salman and 
          Sun, Guolei and Shahbaz Khan, Fahad and Zhu, Fan and Shao, Ling and 
          Xia, Gui-Song and Bai, Xiang}},
  booktitle={{CVPR Workshops}},
  year={{2019}}
}}
"""
    
    with open(isaid_dir / 'info.txt', 'w') as f:
        f.write(info_content)
    print(f"Updated: {isaid_dir / 'info.txt'}")
    
    print("\n" + "="*60)
    print("Tiling complete!")
    print(f"Output directory: {isaid_dir}")
    print(f"  - train/: {len(train_images)} image tiles")
    print(f"  - test/: {len(test_images)} image tiles")
    print(f"  - annotations/train.json: {len(train_annotations)} annotations")
    print(f"  - annotations/test.json: {len(test_annotations)} annotations")
    print(f"  - train_original/: Original large images (backup)")
    print(f"  - test_original/: Original large images (backup)")


if __name__ == '__main__':
    main()
