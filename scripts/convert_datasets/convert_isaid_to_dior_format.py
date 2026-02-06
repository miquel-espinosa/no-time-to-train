#!/usr/bin/env python3
"""
Convert iSAID dataset to DIOR format for the pipeline.

iSAID format:
- train/annotations/iSAID_train.json: COCO format annotations for training
- val/annotations/iSAID_val.json: COCO format annotations for validation
- Images must be obtained separately from DOTA dataset (same image IDs)

IMPORTANT: iSAID uses images from the DOTA-v1.0 dataset. You need to:
1. Download DOTA images from: https://captain-whu.github.io/DOTA/index.html
2. Provide the path to the DOTA images directory using --dota_images_dir

DIOR format output:
- train/: Training images
- test/: Test images (uses iSAID validation set since test has no annotations)
- annotations/train.json: COCO format annotations for training
- annotations/test.json: COCO format annotations for test

iSAID Categories (15 classes):
1. storage_tank, 2. Large_Vehicle, 3. Small_Vehicle, 4. plane, 5. ship,
6. Swimming_pool, 7. Harbor, 8. tennis_court, 9. Ground_Track_Field,
10. Soccer_ball_field, 11. baseball_diamond, 12. Bridge, 13. basketball_court,
14. Roundabout, 15. Helicopter

NOTE: The original iSAID train and val splits have DIFFERENT category ID mappings!
This script remaps all annotations to use a unified canonical category order.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image

# Canonical iSAID category order (matches metainfo.py and train split)
CANONICAL_CATEGORIES = [
    {"id": 1, "name": "storage_tank", "supercategory": "none"},
    {"id": 2, "name": "Large_Vehicle", "supercategory": "none"},
    {"id": 3, "name": "Small_Vehicle", "supercategory": "none"},
    {"id": 4, "name": "plane", "supercategory": "none"},
    {"id": 5, "name": "ship", "supercategory": "none"},
    {"id": 6, "name": "Swimming_pool", "supercategory": "none"},
    {"id": 7, "name": "Harbor", "supercategory": "none"},
    {"id": 8, "name": "tennis_court", "supercategory": "none"},
    {"id": 9, "name": "Ground_Track_Field", "supercategory": "none"},
    {"id": 10, "name": "Soccer_ball_field", "supercategory": "none"},
    {"id": 11, "name": "baseball_diamond", "supercategory": "none"},
    {"id": 12, "name": "Bridge", "supercategory": "none"},
    {"id": 13, "name": "basketball_court", "supercategory": "none"},
    {"id": 14, "name": "Roundabout", "supercategory": "none"},
    {"id": 15, "name": "Helicopter", "supercategory": "none"}
]


def process_split(src_json, src_images_dir, output_img_dir, output_json_path, split_name):
    """Process a split and copy images."""
    
    # Load source JSON
    with open(src_json, 'r') as f:
        data = json.load(f)
    
    print(f"\n=== Processing {split_name} Split ===")
    print(f"Source JSON: {src_json}")
    print(f"Images: {len(data['images'])}, Annotations: {len(data['annotations'])}")
    
    # Build category ID remapping from source to canonical
    source_cats = {cat['id']: cat['name'] for cat in data['categories']}
    canonical_name_to_id = {cat['name']: cat['id'] for cat in CANONICAL_CATEGORIES}
    
    # Create mapping: source_id -> canonical_id
    cat_id_remap = {}
    for src_id, src_name in source_cats.items():
        if src_name in canonical_name_to_id:
            cat_id_remap[src_id] = canonical_name_to_id[src_name]
        else:
            print(f"WARNING: Category '{src_name}' not found in canonical list!")
    
    print(f"Category ID remapping for {split_name}:")
    for src_id, tgt_id in sorted(cat_id_remap.items()):
        src_name = source_cats[src_id]
        print(f"  {src_id} ({src_name}) -> {tgt_id}")
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Track which images are found/missing
    found_images = []
    missing_images = []
    new_image_id_map = {}
    
    # Copy images and build new image list
    new_images = []
    for idx, img_info in enumerate(tqdm(data['images'], desc=f"Processing {split_name} images")):
        src_path = Path(src_images_dir) / img_info['file_name']
        
        if not src_path.exists():
            # Try alternate extensions
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                alt_path = src_path.with_suffix(ext)
                if alt_path.exists():
                    src_path = alt_path
                    break
        
        if src_path.exists():
            dst_path = Path(output_img_dir) / src_path.name
            
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            
            # Get image dimensions - read from file if not in JSON
            width = img_info.get('width', 0)
            height = img_info.get('height', 0)
            if width == 0 or height == 0:
                # Read dimensions from actual image file
                with Image.open(dst_path) as img:
                    width, height = img.size
            
            # Create new image entry with updated ID
            new_id = len(new_images) + 1
            new_image_id_map[img_info['id']] = new_id
            
            new_img_info = {
                'file_name': src_path.name,
                'height': height,
                'width': width,
                'id': new_id
            }
            new_images.append(new_img_info)
            found_images.append(img_info['file_name'])
        else:
            missing_images.append(img_info['file_name'])
    
    if missing_images:
        print(f"\nWarning: {len(missing_images)} images not found in {src_images_dir}")
        if len(missing_images) <= 10:
            for f in missing_images:
                print(f"  Missing: {f}")
        else:
            print(f"  First 10 missing: {missing_images[:10]}")
    
    if not new_images:
        print(f"\nERROR: No images found! Please check --dota_images_dir path.")
        print(f"Expected images like: {data['images'][0]['file_name']}")
        return None
    
    # Filter annotations to only include found images and remap category IDs
    new_annotations = []
    annotation_id = 1
    for ann in tqdm(data['annotations'], desc="Processing annotations"):
        if ann['image_id'] in new_image_id_map:
            new_ann = ann.copy()
            new_ann['id'] = annotation_id
            new_ann['image_id'] = new_image_id_map[ann['image_id']]
            
            # CRITICAL: Remap category_id to canonical order
            if ann['category_id'] in cat_id_remap:
                new_ann['category_id'] = cat_id_remap[ann['category_id']]
            else:
                print(f"WARNING: Annotation {ann['id']} has unknown category_id {ann['category_id']}")
                continue  # Skip this annotation
            
            # Ensure required fields exist
            if 'iscrowd' not in new_ann:
                new_ann['iscrowd'] = 0
            if 'ignore' not in new_ann:
                new_ann['ignore'] = 0
            
            # Remove extra fields not needed in standard COCO format
            new_ann.pop('category_name', None)
            new_ann.pop('ins_file_name', None)
            new_ann.pop('seg_file_name', None)
            
            new_annotations.append(new_ann)
            annotation_id += 1
    
    # Build output COCO dict with canonical categories
    output_data = {
        'images': new_images,
        'type': 'instances',
        'annotations': new_annotations,
        'categories': CANONICAL_CATEGORIES
    }
    
    # Save output JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f)
    
    print(f"\nSaved: {output_json_path}")
    print(f"  Images: {len(new_images)} (of {len(data['images'])} total)")
    print(f"  Annotations: {len(new_annotations)}")
    
    return len(new_images), len(new_annotations)


def main():
    parser = argparse.ArgumentParser(
        description='Convert iSAID dataset to DIOR format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_isaid_to_dior_format.py \\
      --dota_train_images_dir /path/to/DOTA/train/images \\
      --dota_val_images_dir /path/to/DOTA/val/images

Note: iSAID uses images from DOTA-v1.0. Download from:
  https://captain-whu.github.io/DOTA/index.html
  
  iSAID train uses DOTA train images.
  iSAID val uses DOTA val images.
        """
    )
    parser.add_argument('--isaid_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/iSAID',
                        help='Path to iSAID dataset (with original_files subfolder)')
    parser.add_argument('--dota_images_dir', type=str,
                        default=None,
                        help='Path to DOTA images directory (deprecated, use --dota_train_images_dir)')
    parser.add_argument('--dota_train_images_dir', type=str,
                        default=None,
                        help='Path to DOTA train images directory')
    parser.add_argument('--dota_val_images_dir', type=str,
                        default=None,
                        help='Path to DOTA val images directory')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: same as isaid_dir)')
    args = parser.parse_args()
    
    isaid_dir = Path(args.isaid_dir)
    output_dir = Path(args.output_dir) if args.output_dir else isaid_dir
    
    # Handle backward compatibility with --dota_images_dir
    dota_train_dir = Path(args.dota_train_images_dir) if args.dota_train_images_dir else None
    dota_val_dir = Path(args.dota_val_images_dir) if args.dota_val_images_dir else None
    
    if args.dota_images_dir and not dota_train_dir:
        dota_train_dir = Path(args.dota_images_dir)
    
    # Check if at least one DOTA directory is provided
    if not dota_train_dir and not dota_val_dir:
        print("ERROR: Please provide at least one DOTA images directory.")
        print("  --dota_train_images_dir for train images")
        print("  --dota_val_images_dir for val images")
        print("\niSAID requires images from the DOTA-v1.0 dataset.")
        print("Please download from: https://captain-whu.github.io/DOTA/index.html")
        return
    
    # Check directories exist
    if dota_train_dir and not dota_train_dir.exists():
        print(f"ERROR: DOTA train images directory not found: {dota_train_dir}")
        return
    if dota_val_dir and not dota_val_dir.exists():
        print(f"ERROR: DOTA val images directory not found: {dota_val_dir}")
        return
    
    original_files = isaid_dir / 'original_files'
    train_json = original_files / 'train' / 'annotations' / 'iSAID_train.json'
    val_json = original_files / 'val' / 'annotations' / 'iSAID_val.json'
    
    # Check if annotation files exist
    if not train_json.exists():
        print(f"ERROR: Train annotations not found: {train_json}")
        return
    if not val_json.exists():
        print(f"ERROR: Validation annotations not found: {val_json}")
        return
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"iSAID dataset: {isaid_dir}")
    print(f"DOTA train images: {dota_train_dir}")
    print(f"DOTA val images: {dota_val_dir}")
    print(f"Output directory: {output_dir}")
    
    train_result = None
    test_result = None
    
    # Process training split (uses DOTA train images)
    if dota_train_dir:
        dota_train_images = list(dota_train_dir.glob("P*.png")) + list(dota_train_dir.glob("P*.jpg"))
        print(f"\nFound {len(dota_train_images)} DOTA train images")
        
        train_result = process_split(
            train_json, dota_train_dir, train_dir,
            annotations_dir / 'train.json', 'Train'
        )
    else:
        print("\nSkipping train split (no --dota_train_images_dir provided)")
    
    # Process validation as test split (uses DOTA val images)
    if dota_val_dir:
        dota_val_images = list(dota_val_dir.glob("P*.png")) + list(dota_val_dir.glob("P*.jpg"))
        print(f"\nFound {len(dota_val_images)} DOTA val images")
        
        test_result = process_split(
            val_json, dota_val_dir, test_dir,
            annotations_dir / 'test.json', 'Test (from val)'
        )
    else:
        print("\nSkipping test/val split (no --dota_val_images_dir provided)")
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    if train_result:
        print(f"  - annotations/train.json ({train_result[0]} images, {train_result[1]} annotations)")
        print(f"  - train/ ({train_result[0]} images)")
    if test_result:
        print(f"  - annotations/test.json ({test_result[0]} images, {test_result[1]} annotations)")
        print(f"  - test/ ({test_result[0]} images)")
    
    print("\nNote: iSAID validation set was used as test set since the official")
    print("test set does not have public annotations.")


if __name__ == '__main__':
    main()
