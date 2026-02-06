#!/usr/bin/env python3
"""
Convert SSDD (SAR Ship Detection Dataset) PSeg version to DIOR format for the pipeline.

SSDD PSeg format (Official-SSDD-OPEN/PSeg_SSDD/voc_style/):
- JPEGImages_train/: Training images
- JPEGImages_test/: Test images
- Annotations_train/: VOC-style XML annotations with polygon segmentation
- Annotations_test/: VOC-style XML annotations with polygon segmentation

DIOR format output:
- train/: Training images
- test/: Test images  
- annotations/train.json: COCO format annotations for training
- annotations/test.json: COCO format annotations for test

SSDD Categories (1 class):
- ship
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import shutil
from PIL import Image


def parse_polygon_points(segm_element):
    """Parse polygon points from XML segm element."""
    points = []
    for point_elem in segm_element.findall('point'):
        x, y = point_elem.text.split(',')
        points.append((float(x), float(y)))
    return points


def polygon_to_coco_segmentation(points):
    """Convert list of (x,y) tuples to COCO polygon format (flat list)."""
    flat = []
    for x, y in points:
        flat.extend([x, y])
    return [flat]


def polygon_to_bbox(points):
    """Calculate bounding box from polygon points. Returns [x, y, width, height]."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def polygon_area(points):
    """Calculate polygon area using shoelace formula."""
    n = len(points)
    if n < 3:
        return 0
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2


def parse_xml_annotation(xml_path):
    """Parse a VOC-style XML annotation file with polygon segmentation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image info
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        
        # Get segmentation polygon
        segm = obj.find('segm')
        if segm is not None:
            points = parse_polygon_points(segm)
            if len(points) >= 3:  # Valid polygon needs at least 3 points
                objects.append({
                    'name': name,
                    'difficult': difficult,
                    'polygon': points
                })
    
    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def process_split(images_dir, annotations_dir, output_img_dir, output_json_path, split_name):
    """Process a split (train or test) and create COCO format annotations."""
    
    print(f"\n=== Processing {split_name} Split ===")
    
    # Create output image directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Get list of annotation files
    ann_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])
    
    print(f"Found {len(ann_files)} annotation files")
    
    # COCO format structures
    coco_images = []
    coco_annotations = []
    categories = [{'id': 1, 'name': 'ship', 'supercategory': 'none'}]
    
    annotation_id = 1
    
    for img_idx, ann_file in enumerate(tqdm(ann_files, desc=f"Processing {split_name}")):
        image_id = img_idx + 1
        
        # Parse XML annotation
        xml_path = Path(annotations_dir) / ann_file
        ann_data = parse_xml_annotation(xml_path)
        
        # Find and copy image
        img_filename = ann_data['filename']
        src_img_path = Path(images_dir) / img_filename
        
        if not src_img_path.exists():
            # Try alternate extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_path = src_img_path.with_suffix(ext)
                if alt_path.exists():
                    src_img_path = alt_path
                    img_filename = alt_path.name
                    break
        
        if not src_img_path.exists():
            print(f"Warning: Image not found: {src_img_path}")
            continue
        
        # Copy image
        dst_img_path = Path(output_img_dir) / img_filename
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)
        
        # Get actual image dimensions (in case XML is wrong)
        with Image.open(dst_img_path) as img:
            width, height = img.size
        
        # Add image entry
        coco_images.append({
            'file_name': img_filename,
            'height': height,
            'width': width,
            'id': image_id
        })
        
        # Add annotations for each object
        for obj in ann_data['objects']:
            polygon = obj['polygon']
            segmentation = polygon_to_coco_segmentation(polygon)
            bbox = polygon_to_bbox(polygon)
            area = polygon_area(polygon)
            
            coco_annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,  # ship
                'segmentation': segmentation,
                'bbox': bbox,
                'area': area,
                'iscrowd': 0,
                'ignore': obj['difficult']
            })
            annotation_id += 1
    
    # Create COCO format dictionary
    coco_dict = {
        'images': coco_images,
        'type': 'instances',
        'annotations': coco_annotations,
        'categories': categories
    }
    
    # Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f)
    
    print(f"Saved: {output_json_path}")
    print(f"  Images: {len(coco_images)}")
    print(f"  Annotations: {len(coco_annotations)}")
    
    return len(coco_images), len(coco_annotations)


def main():
    parser = argparse.ArgumentParser(description='Convert SSDD PSeg dataset to DIOR format')
    parser.add_argument('--ssdd_dir', type=str, 
                        default='/home/s2254242/projects/no-time-to-train/data/SSDD',
                        help='Path to SSDD dataset root')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Output directory (default: same as ssdd_dir)')
    args = parser.parse_args()
    
    ssdd_dir = Path(args.ssdd_dir)
    output_dir = Path(args.output_dir) if args.output_dir else ssdd_dir
    
    # PSeg_SSDD paths
    pseg_dir = ssdd_dir / 'Official-SSDD-OPEN' / 'PSeg_SSDD' / 'voc_style'
    
    train_images_dir = pseg_dir / 'JPEGImages_train'
    test_images_dir = pseg_dir / 'JPEGImages_test'
    train_annotations_dir = pseg_dir / 'Annotations_train'
    test_annotations_dir = pseg_dir / 'Annotations_test'
    
    # Check if source directories exist
    for d in [train_images_dir, test_images_dir, train_annotations_dir, test_annotations_dir]:
        if not d.exists():
            print(f"ERROR: Directory not found: {d}")
            return
    
    # Create output directories
    annotations_dir = output_dir / 'annotations'
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    os.makedirs(annotations_dir, exist_ok=True)
    
    print(f"SSDD PSeg dataset: {pseg_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process training split
    train_imgs, train_anns = process_split(
        train_images_dir, train_annotations_dir, train_dir,
        annotations_dir / 'train.json', 'Train'
    )
    
    # Process test split
    test_imgs, test_anns = process_split(
        test_images_dir, test_annotations_dir, test_dir,
        annotations_dir / 'test.json', 'Test'
    )
    
    # Create info.txt
    info_path = output_dir / 'info.txt'
    with open(info_path, 'w') as f:
        f.write("""Dataset: SSDD (SAR Ship Detection Dataset)
Number of classes: 1
Class names: ship
Has bounding boxes: Yes
Has instance segmentation masks: Yes
Segmentation format: Polygon (converted from VOC XML to COCO format)

Source: https://github.com/TianwenZhang0825/Official-SSDD
Images from: TerraSAR-X, RadarSat-2, Sentinel-1 SAR satellites

Train split: {} images, {} annotations
Test split: {} images, {} annotations

Citation:
@article{{zhang2021sar,
  title={{SAR Ship Detection Dataset (SSDD): Official Release and Comprehensive Data Analysis}},
  author={{Zhang, Tianwen and Zhang, Xiaoling and Li, Jianwei and Xu, Xiaowo and Wang, Baoyou and Zhan, Xu and Xu, Yanqin and Ke, Xiao and Zeng, Tianjiao and Su, Hao and others}},
  journal={{Remote Sensing}},
  volume={{13}},
  number={{18}},
  pages={{3690}},
  year={{2021}}
}}
""".format(train_imgs, train_anns, test_imgs, test_anns))
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - annotations/train.json ({train_imgs} images, {train_anns} annotations)")
    print(f"  - annotations/test.json ({test_imgs} images, {test_anns} annotations)")
    print(f"  - train/ ({train_imgs} images)")
    print(f"  - test/ ({test_imgs} images)")
    print(f"  - info.txt")


if __name__ == '__main__':
    main()
