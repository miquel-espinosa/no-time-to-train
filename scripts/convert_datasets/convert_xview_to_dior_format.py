#!/usr/bin/env python3
"""
Convert xView dataset to DIOR format with image tiling.

xView format:
- train_images/: Large TIF images (3000-5000px)
- val_images/: Validation images (no labels, ignored)
- xView_train.geojson: GeoJSON annotations with bounds_imcoords

DIOR format output:
- train/: Training image tiles
- test/: Test image tiles
- annotations/train.json: COCO format annotations
- annotations/test.json: COCO format annotations

Key features:
- Tiles large images into smaller chips (configurable size)
- Adjusts bounding boxes for each tile
- Clips boxes at tile boundaries
- Filters out boxes that are too small or mostly outside the tile
- Splits into train/test by original images (not tiles) to avoid data leakage
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np


# xView class mapping: type_id (11-94) -> class index (0-59)
# -1 means the class is not used
XVIEW_CLASS2INDEX = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  # 0-10: not used
    0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,  # 11-26
    12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1,  # 27-40
    23, 24, 25, -1, 26, 27, -1, 28, -1, 29, 30, 31, 32, 33, 34, 35, 36, 37, -1,  # 41-59
    38, 39, 40, 41, 42, 43, 44, 45, -1, -1, -1, -1,  # 60-71
    46, 47, 48, 49, -1, 50, 51, -1, 52, -1, -1, -1,  # 72-83
    53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59  # 84-94
]

# xView class names (60 classes, indices 0-59)
# Note: Spaces replaced with hyphens for shell script compatibility
XVIEW_CLASSES = [
    'Fixed-wing-Aircraft', 'Small-Aircraft', 'Cargo-Plane', 'Helicopter',
    'Passenger-Vehicle', 'Small-Car', 'Bus', 'Pickup-Truck', 'Utility-Truck',
    'Truck', 'Cargo-Truck', 'Truck-w/Box', 'Truck-Tractor', 'Trailer',
    'Truck-w/Flatbed', 'Truck-w/Liquid', 'Crane-Truck', 'Railway-Vehicle',
    'Passenger-Car', 'Cargo-Car', 'Flat-Car', 'Tank-car', 'Locomotive',
    'Maritime-Vessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge',
    'Fishing-Vessel', 'Ferry', 'Yacht', 'Container-Ship', 'Oil-Tanker',
    'Engineering-Vehicle', 'Tower-crane', 'Container-Crane', 'Reach-Stacker',
    'Straddle-Carrier', 'Mobile-Crane', 'Dump-Truck', 'Haul-Truck',
    'Scraper/Tractor', 'Front-loader/Bulldozer', 'Excavator', 'Cement-Mixer',
    'Ground-Grader', 'Hut/Tent', 'Shed', 'Building', 'Aircraft-Hangar',
    'Damaged-Building', 'Facility', 'Construction-Site', 'Vehicle-Lot',
    'Helipad', 'Storage-Tank', 'Shipping-container-lot', 'Shipping-Container',
    'Pylon', 'Tower'
]


def parse_bounds_imcoords(bounds_str):
    """Parse bounds_imcoords string to [x1, y1, x2, y2]."""
    coords = [int(x) for x in bounds_str.split(',')]
    return coords  # [x1, y1, x2, y2]


def load_geojson_annotations(geojson_path):
    """Load and parse xView GeoJSON annotations."""
    print(f"Loading {geojson_path}...")
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    skipped = 0
    
    for feature in tqdm(data['features'], desc="Parsing annotations"):
        props = feature['properties']
        
        # Skip if no bounds
        if not props.get('bounds_imcoords'):
            skipped += 1
            continue
        
        type_id = props['type_id']
        
        # Map to class index
        if type_id >= len(XVIEW_CLASS2INDEX) or XVIEW_CLASS2INDEX[type_id] == -1:
            skipped += 1
            continue
        
        class_idx = XVIEW_CLASS2INDEX[type_id]
        
        # Parse bounding box [x1, y1, x2, y2]
        bbox_xyxy = parse_bounds_imcoords(props['bounds_imcoords'])
        
        annotations_by_image[props['image_id']].append({
            'bbox_xyxy': bbox_xyxy,
            'class_idx': class_idx,
            'feature_id': props.get('feature_id', 0)
        })
    
    print(f"Loaded annotations for {len(annotations_by_image)} images")
    print(f"Skipped {skipped} invalid annotations")
    
    return annotations_by_image


def clip_box_to_tile(bbox_xyxy, tile_x, tile_y, tile_size):
    """
    Clip a bounding box to a tile and convert to tile-relative coordinates.
    
    Args:
        bbox_xyxy: [x1, y1, x2, y2] in original image coordinates
        tile_x, tile_y: Top-left corner of tile in original image
        tile_size: Size of the tile
    
    Returns:
        Clipped bbox in tile coordinates [x1, y1, x2, y2] or None if invalid
    """
    x1, y1, x2, y2 = bbox_xyxy
    
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
    original_area = (x2 - x1) * (y2 - y1)
    clipped_area = (x2_clip - x1_clip) * (y2_clip - y1_clip)
    
    # Skip if less than 30% of original box is in tile
    if original_area > 0 and clipped_area / original_area < 0.3:
        return None
    
    # Skip very small boxes (less than 4x4 pixels)
    if (x2_clip - x1_clip) < 4 or (y2_clip - y1_clip) < 4:
        return None
    
    return [x1_clip, y1_clip, x2_clip, y2_clip]


def xyxy_to_xywh(bbox_xyxy):
    """Convert [x1, y1, x2, y2] to [x, y, width, height]."""
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]


def tile_image_and_annotations(image_path, annotations, tile_size, overlap, output_dir, 
                                image_prefix, start_image_id, start_ann_id):
    """
    Tile a single image and its annotations.
    
    Returns:
        List of (image_info, annotations) tuples
    """
    img = Image.open(image_path)
    img_width, img_height = img.size
    img_array = np.array(img)
    
    stride = tile_size - overlap
    tiles = []
    
    current_image_id = start_image_id
    current_ann_id = start_ann_id
    
    # Generate tile coordinates
    y_positions = list(range(0, img_height - overlap, stride))
    x_positions = list(range(0, img_width - overlap, stride))
    
    # Ensure we cover the entire image
    if y_positions[-1] + tile_size < img_height:
        y_positions.append(img_height - tile_size)
    if x_positions[-1] + tile_size < img_width:
        x_positions.append(img_width - tile_size)
    
    for tile_y in y_positions:
        for tile_x in x_positions:
            # Ensure tile doesn't go outside image
            tile_y = max(0, min(tile_y, img_height - tile_size))
            tile_x = max(0, min(tile_x, img_width - tile_size))
            
            # Check which annotations fall in this tile
            tile_annotations = []
            for ann in annotations:
                clipped_box = clip_box_to_tile(ann['bbox_xyxy'], tile_x, tile_y, tile_size)
                if clipped_box is not None:
                    tile_annotations.append({
                        'id': current_ann_id,
                        'image_id': current_image_id,
                        'category_id': ann['class_idx'] + 1,  # COCO uses 1-indexed categories
                        'bbox': xyxy_to_xywh(clipped_box),
                        'area': (clipped_box[2] - clipped_box[0]) * (clipped_box[3] - clipped_box[1]),
                        'iscrowd': 0,
                        'ignore': 0,
                        'segmentation': []  # bbox only
                    })
                    current_ann_id += 1
            
            # Skip tiles with no annotations
            if not tile_annotations:
                continue
            
            # Extract and save tile
            tile_array = img_array[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
            
            # Handle edge cases where tile might be smaller
            if tile_array.shape[0] < tile_size or tile_array.shape[1] < tile_size:
                # Pad with zeros if needed
                padded = np.zeros((tile_size, tile_size, tile_array.shape[2] if len(tile_array.shape) > 2 else 1), dtype=tile_array.dtype)
                padded[:tile_array.shape[0], :tile_array.shape[1]] = tile_array if len(tile_array.shape) > 2 else tile_array[..., np.newaxis]
                tile_array = padded
            
            tile_filename = f"{image_prefix}_{tile_x}_{tile_y}.jpg"
            tile_path = output_dir / tile_filename
            
            # Convert to PIL and save as JPEG
            if len(tile_array.shape) == 2:
                tile_img = Image.fromarray(tile_array)
            else:
                tile_img = Image.fromarray(tile_array)
            
            # Convert to RGB if needed
            if tile_img.mode != 'RGB':
                tile_img = tile_img.convert('RGB')
            
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


def create_coco_categories():
    """Create COCO categories from xView classes."""
    categories = []
    for i, name in enumerate(XVIEW_CLASSES):
        categories.append({
            'id': i + 1,  # COCO uses 1-indexed
            'name': name.replace(' ', '-'),
            'supercategory': 'none'
        })
    return categories


def main():
    parser = argparse.ArgumentParser(
        description='Convert xView dataset to DIOR format with tiling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_xview_to_dior_format.py --tile_size 512 --overlap 64

This will:
1. Load xView GeoJSON annotations
2. Tile large images into 512x512 chips with 64px overlap
3. Split by original images (80/20) into train/test
4. Output COCO format annotations
        """
    )
    parser.add_argument('--xview_dir', type=str,
                        default='/home/s2254242/projects/no-time-to-train/data/XVIEW',
                        help='Path to xView dataset')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: same as xview_dir)')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Size of image tiles (default: 512)')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between tiles in pixels (default: 64)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of images for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--min_annotations', type=int, default=1,
                        help='Minimum annotations per image to include (default: 1)')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    xview_dir = Path(args.xview_dir)
    output_dir = Path(args.output_dir) if args.output_dir else xview_dir
    
    # Check if original_files subfolder exists
    if (xview_dir / 'original_files').exists():
        src_root = xview_dir / 'original_files'
    else:
        src_root = xview_dir
    
    # Source paths
    train_images_dir = src_root / 'train_images'
    geojson_path = src_root / 'xView_train.geojson'
    
    # Check paths
    if not train_images_dir.exists():
        print(f"ERROR: train_images not found: {train_images_dir}")
        return
    if not geojson_path.exists():
        print(f"ERROR: GeoJSON not found: {geojson_path}")
        return
    
    # Output paths
    output_train_dir = output_dir / 'train'
    output_test_dir = output_dir / 'test'
    annotations_dir = output_dir / 'annotations'
    
    output_train_dir.mkdir(parents=True, exist_ok=True)
    output_test_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"xView dataset: {xview_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap}px")
    print(f"Train/Test split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    
    # Load annotations
    annotations_by_image = load_geojson_annotations(geojson_path)
    
    # Get list of images with annotations
    available_images = []
    for img_file in train_images_dir.glob('*.tif'):
        if img_file.name in annotations_by_image:
            if len(annotations_by_image[img_file.name]) >= args.min_annotations:
                available_images.append(img_file)
    
    print(f"\nImages with annotations: {len(available_images)}")
    
    # Split images into train/test (by original image to avoid data leakage)
    random.shuffle(available_images)
    split_idx = int(len(available_images) * args.train_ratio)
    train_images = available_images[:split_idx]
    test_images = available_images[split_idx:]
    
    print(f"Train images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    
    # Process train images
    print("\n=== Processing Training Images ===")
    train_image_infos = []
    train_annotations = []
    train_image_id = 1
    train_ann_id = 1
    
    for img_path in tqdm(train_images, desc="Tiling train images"):
        img_name = img_path.stem
        anns = annotations_by_image[img_path.name]
        
        tiles, train_image_id, train_ann_id = tile_image_and_annotations(
            img_path, anns, args.tile_size, args.overlap,
            output_train_dir, img_name, train_image_id, train_ann_id
        )
        
        for img_info, tile_anns in tiles:
            train_image_infos.append(img_info)
            train_annotations.extend(tile_anns)
    
    # Process test images
    print("\n=== Processing Test Images ===")
    test_image_infos = []
    test_annotations = []
    test_image_id = 1
    test_ann_id = 1
    
    for img_path in tqdm(test_images, desc="Tiling test images"):
        img_name = img_path.stem
        anns = annotations_by_image[img_path.name]
        
        tiles, test_image_id, test_ann_id = tile_image_and_annotations(
            img_path, anns, args.tile_size, args.overlap,
            output_test_dir, img_name, test_image_id, test_ann_id
        )
        
        for img_info, tile_anns in tiles:
            test_image_infos.append(img_info)
            test_annotations.extend(tile_anns)
    
    # Create COCO format output
    categories = create_coco_categories()
    
    train_coco = {
        'images': train_image_infos,
        'annotations': train_annotations,
        'categories': categories,
        'type': 'instances'
    }
    
    test_coco = {
        'images': test_image_infos,
        'annotations': test_annotations,
        'categories': categories,
        'type': 'instances'
    }
    
    # Save annotations
    print("\n=== Saving Annotations ===")
    with open(annotations_dir / 'train.json', 'w') as f:
        json.dump(train_coco, f)
    print(f"Saved: {annotations_dir / 'train.json'}")
    
    with open(annotations_dir / 'test.json', 'w') as f:
        json.dump(test_coco, f)
    print(f"Saved: {annotations_dir / 'test.json'}")
    
    # Create info.txt
    info_content = f"""Dataset: xView (DIUx xView 2018 Challenge)
Number of classes: 60
Class names: {', '.join(XVIEW_CLASSES)}
Has bounding boxes: Yes
Has instance segmentation masks: No
Segmentation format: None (bbox only)

Source: https://challenge.xviewdataset.org
Images from: WorldView-3 satellite imagery at 0.3m resolution

Original images: {len(available_images)} (tiled from large satellite images)
Tile size: {args.tile_size}x{args.tile_size} pixels
Tile overlap: {args.overlap} pixels

Train split: {len(train_image_infos)} tiles, {len(train_annotations)} annotations
Test split: {len(test_image_infos)} tiles, {len(test_annotations)} annotations

Note: Images split by original source image (not tiles) to prevent data leakage.
Note: Bounding boxes clipped at tile boundaries, small/partial boxes filtered out.

Citation:
@misc{{lam2018xview,
  title={{xView: Objects in Context in Overhead Imagery}},
  author={{Darius Lam and Richard Kuzma and Kevin McGee and Samuel Dooley and Michael Laielli and Matthew Klaric and Yaroslav Bulatov and Brendan McCord}},
  year={{2018}},
  eprint={{1802.07856}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CV}}
}}
"""
    
    with open(output_dir / 'info.txt', 'w') as f:
        f.write(info_content)
    print(f"Saved: {output_dir / 'info.txt'}")
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - train/: {len(train_image_infos)} image tiles")
    print(f"  - test/: {len(test_image_infos)} image tiles")
    print(f"  - annotations/train.json: {len(train_annotations)} annotations")
    print(f"  - annotations/test.json: {len(test_annotations)} annotations")


if __name__ == '__main__':
    main()
