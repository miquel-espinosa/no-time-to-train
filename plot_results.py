import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import numpy as np
import os
from collections import defaultdict
from PIL import Image
import cv2
import math


def load_coco_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def load_gt_annotations(file_path):
    return COCO(file_path)


def rle_to_polygon(rle):
    mask = mask_utils.decode(rle)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        polygon = contour.flatten().tolist()
        if len(polygon) > 4:  # Ignore really small polygons
            polygons.append(polygon)
    return polygons


def get_category_names(coco):
    categories = coco.loadCats(coco.getCatIds())
    return {cat['id']: cat['name'] for cat in categories}


def plot_annotation(ax, ann, color, category_names, is_gt=False):
    if 'segmentation' in ann:
        if isinstance(ann['segmentation'], dict):  # RLE format
            polygons = rle_to_polygon(ann['segmentation'])
        elif isinstance(ann['segmentation'], list):
            polygons = ann['segmentation']

        for polygon in polygons:
            if isinstance(polygon, list):  # Polygon segmentation
                poly = Polygon(np.array(polygon).reshape(-1, 2), fill=False, edgecolor=color, linewidth=1)
                ax.add_patch(poly)

    label = category_names[ann['category_id']]
    if not is_gt:
        label += f": {ann['score']:.2f}"
    # Put the label in the top left corner of the polygon
    min_x = min(polygon[::2])
    min_y = min(polygon[1::2])
    ax.text(min_x, min_y, label, fontsize=8, color='white', bbox=dict(facecolor=color, alpha=0.5))


def get_color_map(annotations):
    category_ids = set(ann['category_id'] for ann in annotations)
    return {cat_id: plt.cm.rainbow(i / max(len(category_ids), 1)) for i, cat_id in enumerate(sorted(category_ids))}


def visualize_annotations(gt_coco, pred_annotations, output_dir, image_dir, threshold=0.9):
    os.makedirs(output_dir, exist_ok=True)

    category_names = get_category_names(gt_coco)

    # Group predicted annotations by image_id
    pred_annotations_by_image = defaultdict(list)
    for ann in pred_annotations:
        if ann['score'] > threshold:
            pred_annotations_by_image[ann['image_id']].append(ann)

    for image_id, image_pred_annotations in pred_annotations_by_image.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Get image info and load the image
        img_info = gt_coco.loadImgs(image_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = np.array(Image.open(img_path))

        # Plot ground truth annotations
        ax1.imshow(img)
        gt_anns = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=image_id))
        gt_color_map = get_color_map(gt_anns)
        for ann in gt_anns:
            color = gt_color_map[ann['category_id']]
            plot_annotation(ax1, ann, color, category_names, is_gt=True)
        ax1.set_title(f"Ground Truth - Image ID: {image_id}")
        ax1.axis('off')

        # Plot predicted annotations
        ax2.imshow(img)
        pred_color_map = get_color_map(image_pred_annotations)
        for ann in image_pred_annotations:
            color = pred_color_map[ann['category_id']]
            plot_annotation(ax2, ann, color, category_names)
        ax2.set_title(f"Predictions (score > {threshold}) - Image ID: {image_id}")
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{image_id}.png"))
        plt.close()


def visualize_annotations_by_category(gt_coco, pred_annotations, output_dir, image_dir, threshold=0.9):
    os.makedirs(output_dir, exist_ok=True)

    category_names = get_category_names(gt_coco)

    # Group predicted annotations by image_id
    pred_annotations_by_image = defaultdict(list)
    for ann in pred_annotations:
        if ann['score'] > threshold:
            pred_annotations_by_image[ann['image_id']].append(ann)

    for image_id, image_pred_annotations in pred_annotations_by_image.items():
        # Get image info and load the image
        img_info = gt_coco.loadImgs(image_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img = np.array(Image.open(img_path))

        # Get all categories present in the predictions
        categories = sorted(set(ann['category_id'] for ann in image_pred_annotations))

        # Calculate grid size
        n_plots = len(categories) + 1  # +1 for the ground truth plot
        n_cols = min(3, n_plots)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot ground truth annotations
        ax = axes[0]
        ax.imshow(img)
        gt_anns = gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=image_id))
        gt_color_map = get_color_map(gt_anns)
        for ann in gt_anns:
            color = gt_color_map[ann['category_id']]
            plot_annotation(ax, ann, color, category_names, is_gt=True)
        ax.set_title(f"Ground Truth - Image ID: {image_id}")
        ax.axis('off')

        # Plot predicted annotations by category
        pred_color_map = get_color_map(image_pred_annotations)
        for i, category_id in enumerate(categories, start=1):
            ax = axes[i]
            ax.imshow(img)
            category_anns = [ann for ann in image_pred_annotations if ann['category_id'] == category_id]
            color = pred_color_map[category_id]
            for ann in category_anns:
                plot_annotation(ax, ann, color, category_names)
            ax.set_title(f"Predictions - {category_names[category_id]}")
            ax.axis('off')

        # Remove any unused subplots
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"by_category_{image_id}.png"))
        plt.close()


def main():
    
    base_folder_path = 'result_analysis'
    folder_name = 'comparison'
    
    gt_coco = load_gt_annotations('./data/coco/annotations/instances_val2017.json')
    pred_annotations = load_coco_results('coco_results_filter.json')
    image_dir = './data/coco/val2017'
    threshold = 0.85  # Set the threshold here
    visualize_annotations(gt_coco, pred_annotations, f'{base_folder_path}/{folder_name}', image_dir, threshold)
    visualize_annotations_by_category(gt_coco, pred_annotations, f'{base_folder_path}/{folder_name}', image_dir, threshold)
    print(f"Annotation comparisons have been visualized and saved in the 'output_comparisons' and 'output_comparisons_by_category' directories.")

if __name__ == "__main__":
    main()
