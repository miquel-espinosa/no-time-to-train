#!/usr/bin/env python3
"""
Script to extract and visualize best/worst airplane and bird examples.
Saves images with and without bounding boxes.
"""

import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# AIRPLANE special examples:
#   Worst: ann_id=1793419, bbox_ap_iou=0.1349, area=0.000080
#   Best:  ann_id=156832, bbox_ap_iou=0.6355, area=0.161514

# BIRD special examples:
#   Worst: ann_id=2229971, bbox_ap_iou=0.0066, area=0.000019
#   Best:  ann_id=39113, bbox_ap_iou=0.4792, area=0.181345

# Configuration
ANNOTATIONS_PATH = Path("./data/coco/annotations/instances_train2017.json")
IMAGES_DIR = Path("./data/coco/train2017")
OUTPUT_DIR = Path("./scripts/paper_figures/airplane_bird")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Examples to extract
EXAMPLES = {
    "airplane_worst": 1793419,
    "airplane_best": 156832,
    "bird_worst": 2229971,
    "bird_best": 39113,
}

# Load COCO annotations
print(f"Loading annotations from {ANNOTATIONS_PATH}...")
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

# Create annotation lookup
ann_lookup = {ann['id']: ann for ann in coco_data['annotations']}
img_lookup = {img['id']: img for img in coco_data['images']}

# Process each example
for example_name, ann_id in EXAMPLES.items():
    print(f"\nProcessing {example_name} (ann_id={ann_id})...")
    
    # Get annotation and image info
    if ann_id not in ann_lookup:
        print(f"  Warning: Annotation {ann_id} not found!")
        continue
    
    ann = ann_lookup[ann_id]
    img_id = ann['image_id']
    
    if img_id not in img_lookup:
        print(f"  Warning: Image {img_id} not found!")
        continue
    
    img_info = img_lookup[img_id]
    img_filename = img_info['file_name']
    img_path = IMAGES_DIR / img_filename
    
    if not img_path.exists():
        print(f"  Warning: Image file {img_path} not found!")
        continue
    
    # Copy original image
    output_path = OUTPUT_DIR / f"{example_name}.jpg"
    shutil.copy(img_path, output_path)
    print(f"  Saved image to {output_path}")
    
    # Create image with bounding box
    img = Image.open(img_path)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    # Get bbox in COCO format [x, y, width, height]
    bbox = ann['bbox']
    x, y, w, h = bbox
    
    # Draw red bounding box
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    # Remove axes
    ax.axis('off')
    
    # Save with bbox
    bbox_output_path = OUTPUT_DIR / f"{example_name}_bbox.jpg"
    plt.tight_layout(pad=0)
    plt.savefig(bbox_output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  Saved bbox image to {bbox_output_path}")

print("\nDone! All images saved to", OUTPUT_DIR)
