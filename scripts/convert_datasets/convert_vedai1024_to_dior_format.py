#!/usr/bin/env python3
"""
Convert VEDAI1024 dataset to DIOR format for the pipeline.

VEDAI1024 format:
- Vehicules1024/: Contains {id}_co.png (color) and {id}_ir.png (infrared) images
- Annotations1024/: Contains per-image annotation files {id}.txt
  Annotation format per line: x_center y_center orientation class_id is_visible is_occluded x1 x2 x3 x4 y1 y2 y3 y4
- Annotations1024/foldXX.txt: Training image IDs for fold XX
- Annotations1024/foldXXtest.txt: Test image IDs for fold XX

Vehicle classes:
- 1: car
- 2: truck
- 4: tractor
- 5: camping car
- 9: van
- 10: other/vehicle
- 11: pick-up
- 23: boat/ship
- 31: plane

DIOR format output:
- train/: Training images
- test/: Test images
- annotations/train.json: COCO format annotations for training
- annotations/test.json: COCO format annotations for test

Note: This dataset has oriented bounding boxes. We convert the 4 corners to axis-aligned
bounding boxes for COCO format. No segmentation masks are provided (bbox only).
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# Vehicle class mapping
VEDAI_CLASSES = {
    1: 'car',
    2: 'truck',
    4: 'tractor',
    5: 'camping_car',
    9: 'van',
    10: 'other_vehicle',
    11: 'pickup',
    23: 'boat',
    31: 'plane'
}


def parse_annotation_file(ann_path):
    """
    Parse a VEDAI annotation file.
    
    Returns list of dicts with keys:
    - x_center, y_center: center coordinates
    - orientation: vehicle orientation
    - class_id: vehicle class
    - is_visible: fully visible flag
    - is_occluded: occluded flag
    - corners_x: list of 4 x coordinates
    - corners_y: list of 4 y coordinates
    """
    annotations = []
    
    if not os.path.exists(ann_path):
        return annotations
    
    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 14:
                continue
            
            try:
                ann = {
                    'x_center': float(parts[0]),
                    'y_center': float(parts[1]),
                    'orientation': float(parts[2]),
                    'class_id': int(parts[3]),
                    'is_visible': int(parts[4]),
                    'is_occluded': int(parts[5]),
                    'corners_x': [float(parts[i]) for i in range(6, 10)],
                    'corners_y': [float(parts[i]) for i in range(10, 14)]
                }
                annotations.append(ann)
            except (ValueError, IndexError) as e:
                print(f"Error parsing line in {ann_path}: {line}")
                continue
    
    return annotations


def corners_to_bbox(corners_x, corners_y):
    """Convert 4 corners to axis-aligned bounding box [x, y, width, height]."""
    x_min = min(corners_x)
    x_max = max(corners_x)
    y_min = min(corners_y)
    y_max = max(corners_y)
    
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def load_fold_images(fold_file):
    """Load image IDs from a fold file."""
    image_ids = []
    with open(fold_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                image_ids.append(line)
    return image_ids


def process_single_image(args):
    """Process a single image and its annotation. Designed for parallel execution."""
    image_id, images_dir, annotations_dir, output_img_dir, image_type, coco_image_id = args
    
    result = {
        'image_id': coco_image_id,
        'image_info': None,
        'annotations': [],
        'error': None
    }
    
    try:
        # Image filename
        img_filename = f"{image_id}_{image_type}.png"
        img_path = Path(images_dir) / img_filename
        
        if not img_path.exists():
            result['error'] = f"Image not found: {img_path}"
            return result
        
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Copy image
        new_filename = f"{image_id}.png"
        output_img_path = Path(output_img_dir) / new_filename
        
        if not output_img_path.exists():
            shutil.copy2(img_path, output_img_path)
        
        result['image_info'] = {
            'file_name': new_filename,
            'height': height,
            'width': width,
            'id': coco_image_id
        }
        
        # Parse annotation file
        ann_filename = f"{image_id}.txt"
        ann_path = Path(annotations_dir) / ann_filename
        
        annotations = parse_annotation_file(ann_path)
        
        for ann in annotations:
            # Skip unknown classes
            if ann['class_id'] not in VEDAI_CLASSES:
                continue
            
            # Convert corners to bbox
            bbox = corners_to_bbox(ann['corners_x'], ann['corners_y'])
            area = bbox[2] * bbox[3]
            
            result['annotations'].append({
                'category_id': ann['class_id'],
                'bbox': bbox,
                'area': area,
                'is_visible': ann['is_visible'],
                'is_occluded': ann['is_occluded']
            })
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def process_split(image_ids, images_dir, annotations_dir, output_img_dir, 
                  split_name, image_type='co', num_workers=None):
    """Process a split using parallel workers."""
    
    os.makedirs(output_img_dir, exist_ok=True)
    
    print(f"\nProcessing {split_name} split: {len(image_ids)} images")
    
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)
    
    print(f"Using {num_workers} parallel workers")
    
    # Prepare arguments
    task_args = [
        (img_id, str(images_dir), str(annotations_dir), str(output_img_dir), image_type, idx + 1)
        for idx, img_id in enumerate(image_ids)
    ]
    
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    errors = 0
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, args): args for args in task_args}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split_name}"):
            result = future.result()
            
            if result['error']:
                errors += 1
                continue
            
            if result['image_info']:
                coco_images.append(result['image_info'])
            
            for ann in result['annotations']:
                coco_annotations.append({
                    'id': annotation_id,
                    'image_id': result['image_id'],
                    'category_id': ann['category_id'],
                    'area': ann['area'],
                    'bbox': ann['bbox'],
                    'iscrowd': 0,
                    'ignore': 0
                })
                annotation_id += 1
    
    # Sort images by ID
    coco_images.sort(key=lambda x: x['id'])
    
    if errors > 0:
        print(f"  Errors: {errors} images failed to process")
    
    return coco_images, coco_annotations


def get_all_image_ids(annotations_dir):
    """Get all image IDs from annotation files."""
    image_ids = set()
    
    for f in os.listdir(annotations_dir):
        if f.endswith('.txt') and not f.startswith('fold') and not f.startswith('annotation'):
            image_ids.add(f.replace('.txt', ''))
    
    return sorted(list(image_ids))


def main():
    parser = argparse.ArgumentParser(description='Convert VEDAI1024 dataset to DIOR format')
    parser.add_argument('--vedai_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/VEDAI1024',
                        help='Path to VEDAI1024 dataset')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: reorganize in place)')
    parser.add_argument('--fold', type=int, default=1,
                        help='Which fold to use for train/test split (1-10, default: 1)')
    parser.add_argument('--image_type', type=str, default='co',
                        choices=['co', 'ir'],
                        help='Image type: co (color) or ir (infrared), default: co')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    args = parser.parse_args()
    
    vedai_dir = Path(args.vedai_dir)
    
    # Move original files if reorganizing in place
    if args.output_dir:
        output_dir = Path(args.output_dir)
        original_files = vedai_dir
    else:
        original_files = vedai_dir / 'original_files'
        if not original_files.exists():
            print("Moving original files to original_files/")
            original_files.mkdir(exist_ok=True)
            
            for item in os.listdir(vedai_dir):
                if item != 'original_files':
                    src = vedai_dir / item
                    dst = original_files / item
                    shutil.move(str(src), str(dst))
        
        output_dir = vedai_dir
    
    # Define paths
    images_dir = original_files / 'Vehicules1024'
    annotations_dir = original_files / 'Annotations1024'
    
    # Create output directories
    out_annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(out_annotations_dir, exist_ok=True)
    
    print(f"VEDAI1024 dataset: {original_files}")
    print(f"Output directory: {output_dir}")
    print(f"Using fold {args.fold} for train/test split")
    print(f"Using {args.image_type} (color) images" if args.image_type == 'co' else f"Using {args.image_type} (infrared) images")
    
    # Load fold files
    fold_train_file = annotations_dir / f'fold{args.fold:02d}.txt'
    fold_test_file = annotations_dir / f'fold{args.fold:02d}test.txt'
    
    if fold_train_file.exists() and fold_test_file.exists():
        train_ids = load_fold_images(fold_train_file)
        test_ids = load_fold_images(fold_test_file)
        print(f"\nLoaded fold {args.fold}: {len(train_ids)} train, {len(test_ids)} test images")
    else:
        # Fallback: use all images with 80/20 split
        print(f"\nFold files not found, using 80/20 random split")
        all_ids = get_all_image_ids(annotations_dir)
        np.random.seed(42)
        np.random.shuffle(all_ids)
        split_idx = int(len(all_ids) * 0.8)
        train_ids = all_ids[:split_idx]
        test_ids = all_ids[split_idx:]
    
    # Create COCO categories
    coco_categories = []
    for class_id, class_name in sorted(VEDAI_CLASSES.items()):
        coco_categories.append({
            'supercategory': 'vehicle',
            'id': class_id,
            'name': class_name.replace('_', '-')
        })
    
    print(f"\nCategories ({len(coco_categories)}):")
    for cat in coco_categories:
        print(f"  {cat['id']}: {cat['name']}")
    
    # Process training split
    print("\n=== Processing Training Split ===")
    train_images, train_annotations = process_split(
        train_ids, images_dir, annotations_dir, train_dir,
        "train", image_type=args.image_type, num_workers=args.num_workers
    )
    
    train_coco = {
        'images': train_images,
        'type': 'instances',
        'annotations': train_annotations,
        'categories': coco_categories
    }
    
    train_json_path = out_annotations_dir / 'train.json'
    print(f"\nSaving train annotations to {train_json_path}")
    with open(train_json_path, 'w') as f:
        json.dump(train_coco, f)
    print(f"  Images: {len(train_images)}")
    print(f"  Annotations: {len(train_annotations)}")
    
    # Process test split
    print("\n=== Processing Test Split ===")
    test_images, test_annotations = process_split(
        test_ids, images_dir, annotations_dir, test_dir,
        "test", image_type=args.image_type, num_workers=args.num_workers
    )
    
    test_coco = {
        'images': test_images,
        'type': 'instances',
        'annotations': test_annotations,
        'categories': coco_categories
    }
    
    test_json_path = out_annotations_dir / 'test.json'
    print(f"\nSaving test annotations to {test_json_path}")
    with open(test_json_path, 'w') as f:
        json.dump(test_coco, f)
    print(f"  Images: {len(test_images)}")
    print(f"  Annotations: {len(test_annotations)}")
    
    print("\n" + "="*50)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - original_files/ (original data)")
    print(f"  - annotations/train.json ({len(train_images)} images, {len(train_annotations)} annotations)")
    print(f"  - annotations/test.json ({len(test_images)} images, {len(test_annotations)} annotations)")
    print(f"  - train/ ({len(train_images)} images)")
    print(f"  - test/ ({len(test_images)} images)")
    print(f"\nNote: Using fold {args.fold} for train/test split.")
    print(f"Note: Oriented bounding boxes converted to axis-aligned boxes.")
    print(f"Note: No segmentation masks (bounding boxes only).")


if __name__ == '__main__':
    main()
