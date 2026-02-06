#!/usr/bin/env python3
"""
Tile ROOF dataset images into smaller chips.

This script takes the existing ROOF dataset (with large 10000x10000 images)
and tiles it into smaller images suitable for the pipeline.

Handles RLE segmentation masks by:
1. Decoding RLE to binary mask
2. Extracting tile region
3. Re-encoding to RLE (or converting to polygon if small enough)

Input:
- train/: Large training images (10000x10000)
- test/: Large test images
- val/: Large validation images
- annotations/train.json: COCO format with RLE segmentation
- annotations/test.json
- annotations/val.json

Output:
- train/: Tiled training images
- test/: Tiled test images (combined with val)
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


def rle_decode(rle, shape):
    """Decode RLE to binary mask."""
    if isinstance(rle, dict):
        counts = rle['counts']
        if isinstance(counts, str):
            # Compressed RLE
            from pycocotools import mask as mask_utils
            return mask_utils.decode(rle)
        else:
            # Uncompressed RLE
            mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
            pos = 0
            for i, count in enumerate(counts):
                if i % 2 == 1:  # Odd indices are foreground
                    mask[pos:pos+count] = 1
                pos += count
            return mask.reshape(shape, order='F')
    return None


def rle_encode(mask):
    """Encode binary mask to RLE."""
    from pycocotools import mask as mask_utils
    # Ensure mask is in Fortran order (column-major) as expected by COCO
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
    return rle


def mask_to_polygon(mask):
    """Convert binary mask to polygon (list of vertices)."""
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Need at least 3 points for a polygon
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # At least 3 points (6 coordinates)
                polygons.append(contour)
    
    return polygons


def clip_box_to_tile(bbox_xywh, tile_x, tile_y, tile_size):
    """Clip a bounding box to a tile."""
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
    
    # Check if box is valid
    if x2_clip <= x1_clip or y2_clip <= y1_clip:
        return None
    
    # Calculate preservation ratio
    original_area = w * h
    clipped_w = x2_clip - x1_clip
    clipped_h = y2_clip - y1_clip
    clipped_area = clipped_w * clipped_h
    
    if original_area > 0 and clipped_area / original_area < 0.3:
        return None
    
    if clipped_w < 4 or clipped_h < 4:
        return None
    
    return [float(x1_clip), float(y1_clip), float(clipped_w), float(clipped_h)]


def tile_image_and_annotations(image_path, annotations, img_shape, tile_size, overlap, 
                                output_dir, image_prefix, start_image_id, start_ann_id,
                                use_polygon=False):
    """Tile a single image and its annotations with RLE segmentation support."""
    
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    stride = tile_size - overlap
    tiles = []
    
    current_image_id = start_image_id
    current_ann_id = start_ann_id
    
    # Pre-decode all RLE masks for this image
    decoded_masks = {}
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            try:
                mask = rle_decode(ann['segmentation'], (img_height, img_width))
                if mask is not None:
                    decoded_masks[ann['id']] = mask
            except Exception as e:
                pass  # Skip invalid masks
    
    # Generate tile coordinates
    y_positions = list(range(0, max(1, img_height - overlap), stride))
    x_positions = list(range(0, max(1, img_width - overlap), stride))
    
    # Ensure coverage
    if len(y_positions) > 0 and y_positions[-1] + tile_size < img_height:
        y_positions.append(max(0, img_height - tile_size))
    if len(x_positions) > 0 and x_positions[-1] + tile_size < img_width:
        x_positions.append(max(0, img_width - tile_size))
    
    for tile_y in y_positions:
        for tile_x in x_positions:
            tile_y = max(0, min(tile_y, max(0, img_height - tile_size)))
            tile_x = max(0, min(tile_x, max(0, img_width - tile_size)))
            
            # Check which annotations fall in this tile
            tile_annotations = []
            for ann in annotations:
                clipped_box = clip_box_to_tile(ann['bbox'], tile_x, tile_y, tile_size)
                if clipped_box is None:
                    continue
                
                # Handle segmentation
                tile_seg = []
                if ann['id'] in decoded_masks:
                    # Extract tile region from mask
                    full_mask = decoded_masks[ann['id']]
                    tile_mask = full_mask[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
                    
                    # Check if mask has any content in this tile
                    if tile_mask.sum() < 10:  # Skip very small masks
                        continue
                    
                    # Pad if needed
                    if tile_mask.shape[0] < tile_size or tile_mask.shape[1] < tile_size:
                        padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
                        padded[:tile_mask.shape[0], :tile_mask.shape[1]] = tile_mask
                        tile_mask = padded
                    
                    if use_polygon:
                        tile_seg = mask_to_polygon(tile_mask)
                        if not tile_seg:
                            continue
                    else:
                        tile_seg = rle_encode(tile_mask)
                    
                    # Recalculate area from mask
                    area = float(tile_mask.sum())
                else:
                    # No mask, use bbox area
                    area = clipped_box[2] * clipped_box[3]
                
                tile_annotations.append({
                    'id': current_ann_id,
                    'image_id': current_image_id,
                    'category_id': ann['category_id'],
                    'bbox': clipped_box,
                    'area': area,
                    'iscrowd': ann.get('iscrowd', 0),
                    'ignore': ann.get('ignore', 0),
                    'segmentation': tile_seg
                })
                current_ann_id += 1
            
            # Skip tiles with no annotations
            if not tile_annotations:
                continue
            
            # Extract tile
            actual_tile_w = min(tile_size, img_width - tile_x)
            actual_tile_h = min(tile_size, img_height - tile_y)
            tile_img = img.crop((tile_x, tile_y, tile_x + actual_tile_w, tile_y + actual_tile_h))
            
            # Pad if needed
            if actual_tile_w < tile_size or actual_tile_h < tile_size:
                padded = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                padded.paste(tile_img, (0, 0))
                tile_img = padded
            
            if tile_img.mode != 'RGB':
                tile_img = tile_img.convert('RGB')
            
            tile_filename = f"{image_prefix}_{tile_x}_{tile_y}.jpg"
            tile_path = output_dir / tile_filename
            tile_img.save(tile_path, 'JPEG', quality=95)
            
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
                  split_name, use_polygon=False):
    """Process a split: tile images and update annotations."""
    
    print(f"\nLoading {input_json}...")
    with open(input_json, 'r') as f:
        coco_data = json.load(f)
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    
    img_id_to_info = {img['id']: img for img in coco_data['images']}
    
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
        
        img_anns = anns_by_image.get(img_id, [])
        img_shape = (img_info['height'], img_info['width'])
        
        image_prefix = Path(img_filename).stem
        tiles, current_image_id, current_ann_id = tile_image_and_annotations(
            img_path, img_anns, img_shape, tile_size, overlap,
            output_img_dir, image_prefix, current_image_id, current_ann_id,
            use_polygon=use_polygon
        )
        
        for img_info_tile, tile_anns in tiles:
            all_image_infos.append(img_info_tile)
            all_annotations.extend(tile_anns)
    
    return all_image_infos, all_annotations, coco_data['categories']


def main():
    parser = argparse.ArgumentParser(
        description='Tile ROOF dataset into smaller images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--roof_dir', type=str,
                        default='/home/s2254242/projects/no-time-to-train/data/ROOF',
                        help='Path to ROOF dataset')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Size of image tiles (default: 512)')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between tiles in pixels (default: 64)')
    parser.add_argument('--use_polygon', action='store_true',
                        help='Convert RLE to polygon segmentation (default: keep RLE)')
    args = parser.parse_args()
    
    roof_dir = Path(args.roof_dir)
    
    # Input paths
    train_img_dir = roof_dir / 'train'
    test_img_dir = roof_dir / 'test'
    val_img_dir = roof_dir / 'val'
    annotations_dir = roof_dir / 'annotations'
    
    train_json = annotations_dir / 'train.json'
    test_json = annotations_dir / 'test.json'
    val_json = annotations_dir / 'val.json'
    
    # Check paths
    for p in [train_img_dir, train_json]:
        if not p.exists():
            print(f"ERROR: Not found: {p}")
            return
    
    # Output paths
    train_tiled_dir = roof_dir / 'train_tiled'
    test_tiled_dir = roof_dir / 'test_tiled'
    
    print(f"ROOF dataset: {roof_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap}px")
    print(f"Segmentation format: {'Polygon' if args.use_polygon else 'RLE'}")
    
    # Process training split
    train_images, train_annotations, categories = process_split(
        train_json, train_img_dir, train_tiled_dir,
        args.tile_size, args.overlap, "train", args.use_polygon
    )
    
    # Process test and val together as test
    test_images = []
    test_annotations = []
    
    if test_json.exists() and test_img_dir.exists():
        t_images, t_annotations, _ = process_split(
            test_json, test_img_dir, test_tiled_dir,
            args.tile_size, args.overlap, "test", args.use_polygon
        )
        test_images.extend(t_images)
        test_annotations.extend(t_annotations)
    
    if val_json.exists() and val_img_dir.exists():
        # Adjust IDs for val
        max_img_id = max([img['id'] for img in test_images]) if test_images else 0
        max_ann_id = max([ann['id'] for ann in test_annotations]) if test_annotations else 0
        
        v_images, v_annotations, _ = process_split(
            val_json, val_img_dir, test_tiled_dir,
            args.tile_size, args.overlap, "val", args.use_polygon
        )
        
        for img in v_images:
            img['id'] += max_img_id
            test_images.append(img)
        
        for ann in v_annotations:
            ann['id'] += max_ann_id
            ann['image_id'] += max_img_id
            test_annotations.append(ann)
    
    # Backup existing annotations
    import shutil
    for json_file in ['train.json', 'test.json', 'val.json']:
        src = annotations_dir / json_file
        if src.exists():
            backup = annotations_dir / f'{json_file}.backup'
            if not backup.exists():
                shutil.copy2(src, backup)
                print(f"Backed up {src}")
    
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
    
    # Reorganize directories
    print("\n=== Reorganizing directories ===")
    
    old_train = roof_dir / 'train_original'
    old_test = roof_dir / 'test_original'
    old_val = roof_dir / 'val_original'
    
    if not old_train.exists() and train_img_dir.exists():
        train_img_dir.rename(old_train)
        print(f"Renamed {train_img_dir} -> {old_train}")
    
    if not old_test.exists() and test_img_dir.exists():
        test_img_dir.rename(old_test)
        print(f"Renamed {test_img_dir} -> {old_test}")
    
    if not old_val.exists() and val_img_dir.exists():
        val_img_dir.rename(old_val)
        print(f"Renamed {val_img_dir} -> {old_val}")
    
    if train_tiled_dir.exists():
        train_tiled_dir.rename(train_img_dir)
        print(f"Renamed {train_tiled_dir} -> {train_img_dir}")
    
    if test_tiled_dir.exists():
        test_tiled_dir.rename(test_img_dir)
        print(f"Renamed {test_tiled_dir} -> {test_img_dir}")
    
    # Update info.txt
    seg_format = 'Polygon' if args.use_polygon else 'RLE'
    info_content = f"""Dataset: ROOF (Rooftop Instance Segmentation)
Number of classes: 1
Class names: roof
Has bounding boxes: Yes
Has instance segmentation masks: Yes
Segmentation format: {seg_format} (tiled from original 10000x10000 images)

Source: Satellite imagery rooftop detection dataset

Original images: 10000x10000 pixels, tiled into {args.tile_size}x{args.tile_size} chips
Tile size: {args.tile_size}x{args.tile_size} pixels
Tile overlap: {args.overlap} pixels

Train split: {len(train_images)} tiles, {len(train_annotations)} annotations
Test split: {len(test_images)} tiles, {len(test_annotations)} annotations

Note: Original large images backed up in train_original/, test_original/, val_original/
Note: Segmentation masks clipped at tile boundaries.

Citation:
See dataset source for citation information.
"""
    
    with open(roof_dir / 'info.txt', 'w') as f:
        f.write(info_content)
    print(f"Updated: {roof_dir / 'info.txt'}")
    
    print("\n" + "="*60)
    print("Tiling complete!")
    print(f"Output directory: {roof_dir}")
    print(f"  - train/: {len(train_images)} image tiles")
    print(f"  - test/: {len(test_images)} image tiles")
    print(f"  - annotations/train.json: {len(train_annotations)} annotations")
    print(f"  - annotations/test.json: {len(test_annotations)} annotations")


if __name__ == '__main__':
    main()
