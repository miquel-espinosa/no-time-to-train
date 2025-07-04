import pickle
import random
import argparse
import cv2
import os
import numpy as np

from collections import OrderedDict
from pycocotools.coco import COCO

from dev_hongyi.dataset.data_utils import is_valid_annotation

from dev_hongyi.dataset.metainfo import METAINFO


def sample_memory_dataset(json_file, out_path, memory_length, remove_bad, cat_names=[], dataset='coco', allow_duplicates=False, allow_invalid=False):
    coco = COCO(json_file)
    if dataset == 'lvis':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis'])
    elif dataset == 'lvis_common':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis_common'])
    elif dataset == 'lvis_frequent':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis_frequent'])
    elif dataset == 'lvis_rare':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis_rare'])
    elif dataset == 'lvis_minival':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis_minival'])
    elif dataset == 'lvis_minival_common':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis_minival_common'])
    elif dataset == 'lvis_minival_frequent':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis_minival_frequent'])
    elif dataset == 'lvis_minival_rare':
        cat_ids = coco.getCatIds(catNms=METAINFO['lvis_minival_rare'])
    elif dataset == 'pascal_voc_split_1':
        cat_ids = coco.getCatIds(catNms=METAINFO['pascal_voc_split_1'])
    elif dataset == 'pascal_voc_split_2':
        cat_ids = coco.getCatIds(catNms=METAINFO['pascal_voc_split_2'])
    elif dataset == 'pascal_voc_split_3':
        cat_ids = coco.getCatIds(catNms=METAINFO['pascal_voc_split_3'])
    else:
        cat_ids = coco.getCatIds(catNms=METAINFO['default_classes'])

    cat_to_imgs_and_anns = {}
    for ann_id, ann in coco.anns.items():
        if ann["category_id"] not in cat_ids:
            continue
        if 'isimpossible' in ann.keys():
            if remove_bad and ann["isimpossible"] == 1:
                continue
        ann_cat_id = ann["category_id"]
        if ann_cat_id not in cat_to_imgs_and_anns.keys():
            cat_to_imgs_and_anns[ann_cat_id] = []
        cat_to_imgs_and_anns[ann_cat_id].append((ann['image_id'], ann_id))

    sampled_data_by_cat = OrderedDict()
    for cat_id, cat_data in cat_to_imgs_and_anns.items():
        sampled_data_by_cat[cat_id] = []
        sampled_img_ids_cat = []
        invalid_annotations = []
        random.shuffle(cat_data)
        for i in range(len(cat_data)):
            img_id, ann_id = cat_data[i]
            img_info = coco.loadImgs([img_id])[0]
            if not is_valid_annotation(coco.loadAnns([ann_id])[0], img_info):
                if allow_invalid:
                    invalid_annotations.append({'img_id': img_id, 'ann_ids': [ann_id]})
                continue
            if img_id in sampled_img_ids_cat:
                continue
            sampled_img_ids_cat.append(img_id)
            sampled_data_by_cat[cat_id].append({'img_id': img_id, 'ann_ids': [ann_id]})
            if len(sampled_img_ids_cat) >= memory_length:
                break
        if len(sampled_img_ids_cat) < memory_length:
            
            if len(sampled_data_by_cat[cat_id]) == 0 and allow_invalid:
                print("Warning: Class %d has no valid samples. But has %d invalid samples. We allow invalid samples." % (cat_id, len(invalid_annotations)))
                sampled_data_by_cat[cat_id] = invalid_annotations[:memory_length]
            if allow_duplicates:
                needed_samples = memory_length - len(sampled_data_by_cat[cat_id])
                print("Warning: Class %d has less than %d samples. We need %d more samples." % (cat_id, memory_length, needed_samples))
                # Allow duplicates. Add first, then second, until we have enough.
                for i in range(needed_samples):
                    sampled_data_by_cat[cat_id].append(sampled_data_by_cat[cat_id][i])
            else:
                raise ValueError("Reference for class %d is not enough" % cat_id)

    with open(out_path, 'wb') as fw:
        pickle.dump(sampled_data_by_cat, fw)
    print("Results output to: %s" % out_path)

def visualize_image_mask_pair(image, mask, output_path):
    """
    Create a side-by-side visualization of image and mask.
    Args:
        image: RGB image array
        mask: Binary mask array
        output_path: Path to save the visualization
    """
    # Ensure mask is binary and convert to uint8
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Convert single channel mask to 3 channels
    mask_rgb = np.stack([mask, mask, mask], axis=2)
    
    # Ensure image and mask have same height
    h1, w1 = image.shape[:2]
    h2, w2 = mask_rgb.shape[:2]
    target_height = max(h1, h2)
    
    # Resize if necessary
    if h1 != target_height:
        image = cv2.resize(image, (int(w1 * target_height / h1), target_height))
    if h2 != target_height:
        mask_rgb = cv2.resize(mask_rgb, (int(w2 * target_height / h2), target_height))
    
    # Concatenate horizontally
    visualization = np.concatenate([image, mask_rgb], axis=1)
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

def visualize_all_annotations(image, annotations, coco, output_path, seed):
    """
    Create a visualization of an image with all its COCO annotations.
    Args:
        image: RGB image array
        annotations: List of annotation dictionaries
        coco: COCO object
        output_path: Path to save the visualization
    """
    # Create a copy of the image to draw on
    vis_image = image.copy()
    
    # Use a deterministic random state for colors
    rng = np.random.RandomState(seed)
    colors = rng.randint(0, 255, size=(len(annotations), 3))
    
    # Draw each annotation
    for i, ann in enumerate(annotations):
        # Get binary mask
        mask = coco.annToMask(ann)
        color = colors[i]
        
        # Apply the colored mask to the image
        colored_mask = np.zeros_like(image)
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        # Blend the mask with the image
        alpha = 0.5  # transparency
        mask_area = (mask > 0)
        vis_image[mask_area] = vis_image[mask_area] * (1 - alpha) + colored_mask[mask_area] * alpha
        
        # Draw the outline
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, color.tolist(), 2)
    
    # Save the visualization
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

def visualize_memory_dataset(pkl_path, json_file, img_dir, output_dir, seed):
    """
    Visualize the contents of an existing memory dataset pkl file.
    
    Args:
        pkl_path: Path to the existing pkl file containing the sampled data
        json_file: Path to the original COCO json annotations file
        output_dir: Directory where visualizations will be saved
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pkl file
    with open(pkl_path, 'rb') as f:
        sampled_data_by_cat = pickle.load(f)
    
    # Initialize COCO
    coco = COCO(json_file)
    
    # For each category in the sampled data
    for cat_id, samples in sampled_data_by_cat.items():
        for i, sample in enumerate(samples):
            img_id = sample['img_id']
            
            # Load image
            img_info = coco.loadImgs([img_id])[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create selected mask visualization
            mask = np.zeros((img_info['height'], img_info['width']))
            for ann_id in sample['ann_ids']:
                ann_mask = coco.annToMask(coco.loadAnns([ann_id])[0])
                mask = np.logical_or(mask, ann_mask)
            
            # Save mask visualization
            viz_path = os.path.join(output_dir, f'{cat_id}_v{i}_{img_id}_mask.png')
            visualize_image_mask_pair(image, mask, viz_path)
            
            # Get all annotations for this image and create visualization
            all_anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            all_anns_viz_path = os.path.join(output_dir, f'{cat_id}_v{i}_{img_id}_all_anns.png')
            visualize_all_annotations(image, all_anns, coco, all_anns_viz_path, seed)
    
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample few-shot memory dataset')
    parser.add_argument('--n-shot', type=int, required=True, help='Number of shots (memory length) to sample')
    parser.add_argument('--out-path', type=str, required=True, help='Output path for the sampled dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset to sample from')
    parser.add_argument('--plot', action='store_true', help='Plot the sampled dataset')
    parser.add_argument('--img-dir', type=str, default=None, help='Image directory')
    args = parser.parse_args()
    
    if args.img_dir is None and args.plot:
        raise ValueError("Image directory is required for plotting")

    # Set random seed for reproducibility
    random.seed(args.seed)

    if args.dataset == 'coco':
        all_refs_json_file = "./data/coco/annotations/instances_train2017.json"
        sample_memory_dataset(all_refs_json_file, args.out_path, args.n_shot, remove_bad=True, dataset=args.dataset)
    elif args.dataset == 'lvis' or args.dataset == 'lvis_common' or args.dataset == 'lvis_frequent' or args.dataset == 'lvis_rare' \
            or args.dataset == 'lvis_minival' or args.dataset == 'lvis_minival_common' or args.dataset == 'lvis_minival_frequent' \
            or args.dataset == 'lvis_minival_rare':
        all_refs_json_file = "./data/lvis/lvis_v1_train.json"
        sample_memory_dataset(all_refs_json_file, args.out_path, args.n_shot, remove_bad=False, dataset=args.dataset, allow_duplicates=True, allow_invalid=True)
    elif args.dataset == 'pascal_voc_split_1' or args.dataset == 'pascal_voc_split_2' or args.dataset == 'pascal_voc_split_3':
        all_refs_json_file = "./data/pascal_voc/annotations/voc0712_trainval_with_segm.json"
        sample_memory_dataset(all_refs_json_file, args.out_path, args.n_shot, remove_bad=True, dataset=args.dataset)
    else:
        raise ValueError("Invalid dataset: %s" % args.dataset)

    if args.plot:
        visualize_memory_dataset(args.out_path, all_refs_json_file, args.img_dir, args.out_path.replace('.pkl', '_viz'), args.seed)
