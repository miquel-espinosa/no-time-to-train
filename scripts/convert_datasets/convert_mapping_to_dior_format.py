#!/usr/bin/env python3
"""
Convert MAPPING Challenge dataset to DIOR format for the pipeline.

MAPPING Challenge format:
- train/images/: Training images
- train/annotations/annotation_non_augmented.json: COCO format annotations
- val/images/: Validation images  
- val/annotations/annotation_non_augmented.json: COCO format annotations
- test/images/: Test images
- test/annotations/annotation_non_augmented.json: COCO format annotations

DIOR format output:
- train/: Training images (symlinks to avoid copying large dataset)
- test/: Test images (uses val set, symlinks)
- annotations/train.json: COCO format annotations (bbox only, no segmentation)
- annotations/test.json: COCO format annotations (bbox only, no segmentation)

MAPPING Categories (1 class):
- building

Note: Original dataset has polygon segmentation, but we convert to bbox-only
for object detection evaluation.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def compute_bbox_from_segmentation(segmentation):
    """Compute COCO-format bbox [x, y, width, height] from polygon segmentation."""
    if not segmentation or not segmentation[0]:
        return None
    
    # Flatten all polygons and extract x, y coordinates
    all_x = []
    all_y = []
    for poly in segmentation:
        all_x.extend(poly[0::2])  # x coordinates at even indices
        all_y.extend(poly[1::2])  # y coordinates at odd indices
    
    if not all_x or not all_y:
        return None
    
    min_x = min(all_x)
    min_y = min(all_y)
    max_x = max(all_x)
    max_y = max(all_y)
    
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def compute_area_from_segmentation(segmentation):
    """Compute area using shoelace formula for polygon."""
    if not segmentation or not segmentation[0]:
        return 0
    
    total_area = 0
    for poly in segmentation:
        xs = poly[0::2]
        ys = poly[1::2]
        n = len(xs)
        if n < 3:
            continue
        # Shoelace formula
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += xs[i] * ys[j]
            area -= xs[j] * ys[i]
        total_area += abs(area) / 2
    
    return total_area


def convert_annotations(src_json, output_json_path, bbox_only=True):
    """Convert annotations, optionally removing segmentation."""
    
    # Load source JSON
    with open(src_json, 'r') as f:
        data = json.load(f)
    
    print(f"Source: {len(data['images'])} images, {len(data['annotations'])} annotations")
    
    # Fix category ID (original uses 100, we use 1 for consistency)
    categories = [{'id': 1, 'name': 'building', 'supercategory': 'none'}]
    
    # Process annotations - recalculate bbox from segmentation
    # (original bboxes are in non-standard format)
    new_annotations = []
    skipped = 0
    for ann in tqdm(data['annotations'], desc="Processing annotations"):
        segmentation = ann.get('segmentation', [])
        
        # Recalculate bbox from segmentation (original bboxes are wrong)
        bbox = compute_bbox_from_segmentation(segmentation)
        if bbox is None:
            skipped += 1
            continue
        
        # Recalculate area as well
        area = compute_area_from_segmentation(segmentation)
        
        new_ann = {
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': 1,  # Remap from 100 to 1
            'bbox': bbox,
            'area': area,
            'iscrowd': ann.get('iscrowd', 0),
            'ignore': 0
        }
        
        if bbox_only:
            new_ann['segmentation'] = []
        else:
            new_ann['segmentation'] = segmentation
        
        new_annotations.append(new_ann)
    
    if skipped > 0:
        print(f"  Skipped {skipped} annotations with invalid segmentation")
    
    # Process images - ensure proper format
    new_images = []
    for img in data['images']:
        new_images.append({
            'id': img['id'],
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        })
    
    # Create output COCO dict
    output_data = {
        'images': new_images,
        'type': 'instances',
        'annotations': new_annotations,
        'categories': categories
    }
    
    # Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Saved: {output_json_path}")
    print(f"  Images: {len(new_images)}")
    print(f"  Annotations: {len(new_annotations)}")
    
    return len(new_images), len(new_annotations)


def create_image_symlinks(src_images_dir, dst_images_dir):
    """Create symlinks to images instead of copying (saves disk space)."""
    
    os.makedirs(dst_images_dir, exist_ok=True)
    
    src_path = Path(src_images_dir).resolve()
    dst_path = Path(dst_images_dir).resolve()
    
    # Check if already linked
    if dst_path.is_symlink():
        print(f"Symlink already exists: {dst_path}")
        return
    
    # If directory exists with files, check if it's the same
    if dst_path.exists() and any(dst_path.iterdir()):
        print(f"Directory already has files: {dst_path}")
        return
    
    # Remove empty directory if exists
    if dst_path.exists():
        dst_path.rmdir()
    
    # Create symlink
    os.symlink(src_path, dst_path)
    print(f"Created symlink: {dst_path} -> {src_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert MAPPING dataset to DIOR format')
    parser.add_argument('--mapping_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/MAPPING',
                        help='Path to MAPPING dataset root')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: same as mapping_dir)')
    parser.add_argument('--use_symlinks', action='store_true', default=True,
                        help='Use symlinks for images instead of copying (default: True)')
    parser.add_argument('--copy_images', action='store_true',
                        help='Copy images instead of using symlinks')
    parser.add_argument('--keep_segmentation', action='store_true',
                        help='Keep segmentation masks (default: bbox only)')
    args = parser.parse_args()
    
    mapping_dir = Path(args.mapping_dir)
    output_dir = Path(args.output_dir) if args.output_dir else mapping_dir
    
    use_symlinks = not args.copy_images
    bbox_only = not args.keep_segmentation
    
    # Source paths - check if original_files subfolder exists (reorganized structure)
    if (mapping_dir / 'original_files').exists():
        src_root = mapping_dir / 'original_files'
    else:
        src_root = mapping_dir
    
    train_images_dir = src_root / 'train' / 'images'
    val_images_dir = src_root / 'val' / 'images'
    train_ann_json = src_root / 'train' / 'annotations' / 'annotation_non_augmented.json'
    val_ann_json = src_root / 'val' / 'annotations' / 'annotation_non_augmented.json'
    
    # Check if source directories exist
    for p in [train_images_dir, val_images_dir, train_ann_json, val_ann_json]:
        if not p.exists():
            print(f"ERROR: Not found: {p}")
            return
    
    # Output paths
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train_images'  # Use different name to avoid conflict
    test_dir = output_dir / 'test_images'
    
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"MAPPING dataset: {mapping_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'Symlinks' if use_symlinks else 'Copy'}")
    print(f"Annotations: {'BBox only' if bbox_only else 'With segmentation'}")
    
    # Process training annotations
    print("\n=== Processing Train Split ===")
    train_imgs, train_anns = convert_annotations(
        train_ann_json,
        annotations_dir / 'train.json',
        bbox_only=bbox_only
    )
    
    # Process validation as test (since we use val for testing)
    print("\n=== Processing Test Split (from val) ===")
    test_imgs, test_anns = convert_annotations(
        val_ann_json,
        annotations_dir / 'test.json',
        bbox_only=bbox_only
    )
    
    # Handle images
    print("\n=== Setting up image directories ===")
    if use_symlinks:
        create_image_symlinks(train_images_dir, train_dir)
        create_image_symlinks(val_images_dir, test_dir)
    else:
        print("Copying images (this may take a while)...")
        if not train_dir.exists():
            shutil.copytree(train_images_dir, train_dir)
        if not test_dir.exists():
            shutil.copytree(val_images_dir, test_dir)
    
    # Create info.txt
    info_path = output_dir / 'info.txt'
    with open(info_path, 'w') as f:
        f.write(f"""Dataset: MAPPING Challenge (CrowdAI Building Footprint Segmentation)
Number of classes: 1
Class names: building
Has bounding boxes: Yes
Has instance segmentation masks: {'Yes' if not bbox_only else 'No (bbox only)'}
Segmentation format: {'Polygon (COCO format)' if not bbox_only else 'None (converted to bbox only)'}

Source: https://www.aicrowd.com/challenges/mapping-challenge
Images from: Satellite imagery for building footprint detection

Train split: {train_imgs} images, {train_anns} annotations
Test split: {test_imgs} images, {test_anns} annotations (from validation set)

Note: Using non-augmented version of the dataset.
Note: Images are {'symlinked' if use_symlinks else 'copied'} from original location.

Citation:
@misc{{mapping_challenge,
  title={{Mapping Challenge}},
  author={{CrowdAI}},
  year={{2018}},
  howpublished={{\\url{{https://www.aicrowd.com/challenges/mapping-challenge}}}}
}}
""")
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - annotations/train.json ({train_imgs} images, {train_anns} annotations)")
    print(f"  - annotations/test.json ({test_imgs} images, {test_anns} annotations)")
    print(f"  - train_images/ -> {train_images_dir}")
    print(f"  - test_images/ -> {val_images_dir}")
    print(f"  - info.txt")


if __name__ == '__main__':
    main()
