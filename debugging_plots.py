import numpy as np
import yaml
import torch
import os
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from dev_hongyi.utils import print_dict
from dev_hongyi.dataset.coco_ref_dataset import COCORefTestDataset
from dev_hongyi.pl_wrapper.sam2ref_pl import RefSam2LightningModel
from plot_results import visualize_annotations_by_category, visualize_annotations
from collections import defaultdict
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import math
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_configs(config_path):

    configs = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    
    # configs["model"]["init_args"]["dataset_cfgs"]["train"]["root"] = configs["model"]["init_args"]["dataset_cfgs"]["train"]["root"].replace("data", f"data_{machine}")
    # configs["model"]["init_args"]["dataset_cfgs"]["train"]["json_file"] = configs["model"]["init_args"]["dataset_cfgs"]["train"]["json_file"].replace("data", f"data_{machine}")
    # configs["model"]["init_args"]["dataset_cfgs"]["fill_memory"]["root"] = configs["model"]["init_args"]["dataset_cfgs"]["fill_memory"]["root"].replace("data", f"data_{machine}")
    # configs["model"]["init_args"]["dataset_cfgs"]["fill_memory"]["json_file"] = configs["model"]["init_args"]["dataset_cfgs"]["fill_memory"]["json_file"].replace("data", f"data_{machine}")
    # configs["model"]["init_args"]["dataset_cfgs"]["test"]["root"] = configs["model"]["init_args"]["dataset_cfgs"]["test"]["root"].replace("data", f"data_{machine}")
    # configs["model"]["init_args"]["dataset_cfgs"]["test"]["json_file"] = configs["model"]["init_args"]["dataset_cfgs"]["test"]["json_file"].replace("data", f"data_{machine}")
    
    return configs

def print_coco_results(coco, coco_gt, results, img_ids):
    coco_results = coco.loadRes(results)
    coco_gt = coco.loadRes(coco_gt)
    
    cocoEval_bbox = COCOeval(coco_gt, coco_results, "bbox")
    cocoEval_bbox.params.imgIds = img_ids
    cocoEval_bbox.evaluate()
    cocoEval_bbox.accumulate()
    cocoEval_bbox.summarize()

    cocoEval_segm = COCOeval(coco_gt, coco_results, "segm")
    cocoEval_segm.params.imgIds = img_ids
    cocoEval_segm.evaluate()
    cocoEval_segm.accumulate()
    cocoEval_segm.summarize()

def get_args_parser():
    parser = argparse.ArgumentParser("Finetune MLP")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--tmp_ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--batch_sample", type=int, default=0, help="Batch sample to plot")
    parser.add_argument("--return_iou_grid_scores", action="store_true", help="Return IOU grid scores")
    return parser.parse_args()

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

def polygon_to_rle(polygons, height, width):
    """
    Convert COCO polygon segmentation to RLE mask.
    
    Args:
        polygons (list): List of polygons, where each polygon is a list of x,y coordinates
        height (int): Height of the image
        width (int): Width of the image
        
    Returns:
        dict: RLE encoded mask
    """
    # Create an empty binary mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert each polygon to a numpy array and reshape it
    for polygon in polygons:
        # Reshape the polygon to pairs of x,y coordinates
        polygon = np.array(polygon).reshape(-1, 2)
        
        # Convert to integer type
        polygon = polygon.astype(np.int32)
        
        # Draw the polygon on the mask
        cv2.fillPoly(mask, [polygon], 1)
    
    # Convert the binary mask to RLE
    rle = mask_utils.encode(np.asfortranarray(mask))
    
    return rle

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

def visualize_annotations_by_category(gt_coco, pred_annotations, output_dir, image_dir, threshold=0.9, points=None):
    
    catid_to_catind = {k: i for i, k in enumerate(gt_coco.cats.keys())}
    
    if points is not None:
        scores_for_iou_plotting, points_for_iou_plotting = points
    
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
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 7*n_rows))
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
            ax.imshow(img, alpha=0.9)
            category_anns = [ann for ann in image_pred_annotations if ann['category_id'] == category_id]
            color = pred_color_map[category_id]
            for ann in category_anns:
                plot_annotation(ax, ann, color, category_names)
            
            if points is not None:
                # Extract scores and points for the selected category
                cat_scores = scores_for_iou_plotting[catid_to_catind[category_id]].cpu().numpy()
                cat_points = points_for_iou_plotting[catid_to_catind[category_id]].cpu().numpy()
                scaled_points = torch.tensor(cat_points) / 1024 * torch.tensor([img_info['width'], img_info['height']]) 

                # Create a scatter plot
                scatter = ax.scatter(scaled_points[:, 0], scaled_points[:, 1], c=cat_scores, cmap='jet', s=120, alpha=0.75)
                
                # Add colorbar with the same height as the image
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(scatter, cax=cax, label='IOU Score')
                
                # Set x and y limits to image dimensions
                ax.set_xlim(0, img_info['width'])
                ax.set_ylim(img_info['height'], 0)  # Reverse y-axis to match image coordinates

            ax.set_title(f"Predictions - {category_names[category_id]}")
            ax.axis('off')
        
        # Remove any unused subplots
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"by_category_{image_id}.png"))
        plt.close()

def main(args):
    device = torch.device("cuda")    
    
    configs = load_configs(args.config_path)
    model_configs = configs['model']['init_args']
    
    dataset = COCORefTestDataset(
        root=model_configs['dataset_cfgs']['test']['root'],
        json_file=model_configs['dataset_cfgs']['test']['json_file'],
        image_size=model_configs['dataset_cfgs']['test']['image_size'],
        n_points_per_edge=model_configs['dataset_cfgs']['test']['n_points_per_edge']
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,  # always use bs=1 for eval
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda batch: batch
    )
    
    model = RefSam2LightningModel.load_from_checkpoint(args.tmp_ckpt, **model_configs, test_mode='test', weights_only=True).to(device)
    
    model = model.to(device)
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")
    
    model.eval()
    selected_batch = None
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == args.batch_sample:
            selected_batch = batch
            break
        
    with torch.no_grad():
        if args.return_iou_grid_scores:
            out = model(selected_batch, return_iou_grid_scores=True)[0]
            scores_for_iou_plotting = out.pop("scores_for_iou_plotting")
            points_for_iou_plotting = out.pop("points_for_iou_plotting")
        else:
            out = model(selected_batch)[0] # Select first because bs=1
        # query_points = batch[0]['query_points']
        print("Filename: ", selected_batch[0]['target_img_info']['file_name'])

    # Load ground truth annotations
    coco = COCO(model_configs['dataset_cfgs']['test']['json_file'])
    ind_to_cat_id = {i: k for i, k in enumerate(coco.cats.keys())}

    # Convert to COCO-style annotations
    processed_annotations = []
    for mask, score, category_id in zip(out['binaay_masks'], out['scores'], out['labels']):
        segmentation = mask_utils.encode(np.asfortranarray(mask.cpu().numpy().astype(np.uint8)))
        segmentation['counts'] = segmentation['counts'].decode("utf-8")
        annotation = {
            'image_id': out['image_info']['id'],
            'category_id': ind_to_cat_id[category_id.item()],
            'segmentation': segmentation,
            'score': score.item()
        }
        processed_annotations.append(annotation)
    
    # Prepare GT annotation for single image
    gt_ann = coco.loadAnns(coco.getAnnIds(imgIds=selected_batch[0]['target_img_info']['id']))[0]
    # segm = mask_utils.encode(np.asfortranarray(np.array(gt_ann['segmentation']).astype(np.uint8)))
    img_info = coco.loadImgs(selected_batch[0]['target_img_info']['id'])[0]
    segm = polygon_to_rle(gt_ann['segmentation'], img_info['height'], img_info['width'])
    segm['counts'] = segm['counts'].decode("utf-8")
    coco_gt = {
        'image_id': gt_ann['image_id'],
        'category_id': gt_ann['category_id'],
        'segmentation': segm,
        'score': 1.0
    }

    print_coco_results(coco, [coco_gt], processed_annotations, img_ids=[selected_batch[0]['target_img_info']['id']])

    # Create output directory
    output_dir = './debugging_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Visualize the results by category
    visualize_annotations_by_category(
        coco,
        processed_annotations,
        output_dir,
        model_configs['dataset_cfgs']['test']['root'],
        threshold=0.5,  # Visualisation threshold
        points=(scores_for_iou_plotting, points_for_iou_plotting) if args.return_iou_grid_scores else None
    )
    
    # Visualize all results
    visualize_annotations(
        coco,
        processed_annotations,
        output_dir,
        model_configs['dataset_cfgs']['test']['root'],
        threshold=0.5,  # Visualisation threshold
    )

    print(f"Visualization saved in {output_dir}")


if __name__ == "__main__":
    args = get_args_parser()
    main(args)

