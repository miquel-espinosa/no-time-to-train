import os
import copy


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from dev_hongyi.dataset.metainfo import METAINFO


def draw_box_on_image(
    image,
    boxes,
    str_list,
    show_label=True,
    colors=None,
    thickness=2,
    font_file='./RobotoMono-MediumItalic.ttf'
):
    #default_color = (0, 250, 154)
    default_color = (255, 165, 0)
    draw = ImageDraw.Draw(image)
    
    # Calculate thickness and font size based on image width
    img_width = image.size[0]
    thickness = max(2, int(img_width * 0.003))  # Scale thickness with image width
    font_size = int(img_width * 0.02)  # Scale font size with image width
    font = ImageFont.truetype(font_file, size=font_size)

    for i in range(len(boxes)):
        if colors is not None:
            color = tuple([int(c) for c in colors[i]])
        else:
            color = default_color
        xmin, ymin, xmax, ymax = boxes[i]
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

        dstr = str_list[i]

        display_str_heights = font.getbbox(dstr)[3]
        total_display_str_height = (1 + 2 * 0.05) * display_str_heights

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        text_width, text_height = font.getbbox(dstr)[2], font.getbbox(dstr)[3]
        margin = np.ceil(0.1 * text_height)
        if show_label:
            draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width+3*margin, text_bottom)], fill=color)
            draw.text((left + margin, text_bottom - text_height - margin), dstr, fill='black', font=font)


def generate_distinct_colors(n_colors, seed=42):
    """Generate n_colors distinct RGB colors using HSV color space.
    
    Args:
        n_colors (int): Number of colors to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of RGB colors as [R,G,B] with values in range [0,255]
    """
    np.random.seed(seed)
    
    # Use HSV color space for more visually distinct colors
    colors = []
    for i in range(n_colors):
        # Vary hue uniformly across the spectrum
        h = i / n_colors
        # Add some random variation to saturation and value, but keep them high for visibility
        s = np.random.uniform(0.6, 1.0)
        v = np.random.uniform(0.7, 1.0)
        
        # Convert HSV to RGB
        rgb = plt.cm.hsv(h)[0:3]  # Get RGB from matplotlib's hsv colormap
        # Scale to 0-255 range and ensure integers
        rgb = [int(x * 255) for x in rgb]
        colors.append(rgb)
    
    return colors


def vis_coco(gt_bboxes, gt_labels, gt_masks, scores, labels, bboxes, masks, score_thr, img_path, out_path, show_scores=False, dataset_name='COCO'):
    # Move template_colors inside the function
    if dataset_name.lower() == 'lvis':
        # Generate 1203 colors for LVIS dataset
        template_colors = generate_distinct_colors(1203)
    else:
        # Original template_colors list for other datasets
        template_colors = [
            [31, 248, 130],   # Bright green
            [72, 157, 163],   # Teal
            [34, 87, 33],     # Dark green 
            [184, 230, 19],   # Yellow-green
            [105, 213, 126],  # Sea green
            [45, 193, 253],   # Sky blue
            [181, 130, 22],   # Brown
            [10, 130, 187],   # Blue
            [126, 107, 95],   # Gray-brown
            [118, 238, 60],   # Lime green
            [54, 206, 204],   # Turquoise
            [217, 25, 86],    # Pink-red
            [191, 128, 38],   # Orange-brown
            [127, 81, 162],   # Purple
            [118, 40, 42],    # Dark red
            [212, 124, 74],   # Salmon
            [213, 120, 92],   # Light brown
            [42, 49, 90],     # Navy blue
            [136, 217, 238],  # Light blue
            [53, 153, 28],    # Forest green
            [201, 159, 141],  # Beige
            [234, 197, 227],  # Light pink
            [241, 94, 89],    # Coral
            [126, 57, 48],    # Maroon
            [250, 194, 210],  # Pink
            [173, 182, 57],   # Olive
            [155, 146, 171],  # Gray-purple
            [18, 188, 67],    # Emerald
            [35, 131, 215],   # Royal blue
            [48, 128, 122],   # Dark teal
            [95, 153, 143],   # Gray-green
            [136, 7, 58],     # Wine red
            [247, 128, 122],  # Light coral
            [160, 231, 230],  # Light cyan
            [111, 44, 41],    # Dark brown
            [21, 152, 70],    # Green
            [150, 26, 230],   # Violet
            [168, 178, 253],  # Periwinkle
            [209, 240, 237],  # Very light cyan
            [89, 82, 75],     # Dark gray
            [193, 205, 105],  # Light olive
            [49, 14, 24],     # Very dark red
            [141, 40, 25],    # Rust
            [127, 175, 187],  # Gray-blue
            [47, 70, 75],     # Dark gray-blue
            [137, 246, 43],   # Bright lime
            [233, 101, 196],  # Hot pink
            [81, 114, 242],   # Cornflower blue
            [239, 101, 19],   # Orange
            [165, 98, 237],   # Medium purple
            [242, 78, 149],   # Deep pink
            [93, 137, 102],   # Sage green
            [186, 136, 77],   # Tan
            [226, 61, 132],   # Dark pink
            [124, 44, 151],   # Deep purple
            [53, 23, 4],      # Very dark brown
            [127, 228, 159],  # Mint green
            [31, 16, 107],    # Indigo
            [119, 0, 39],     # Burgundy
            [103, 129, 194],  # Steel blue
            [62, 121, 156],   # Medium blue
            [2, 190, 15],     # Kelly green
            [207, 179, 13],   # Gold
            [9, 233, 221],    # Bright cyan
            [51, 50, 240],    # Electric blue
            [230, 191, 82],   # Light gold
            [146, 7, 160],    # Magenta
            [51, 243, 58],    # Neon green
            [176, 209, 19],   # Yellow-lime
            [252, 135, 81],   # Light orange
            [242, 58, 62],    # Red
            [100, 173, 98],   # Medium green
            [120, 229, 240],  # Light sky blue
            [124, 69, 112],   # Plum
            [106, 44, 52],    # Dark burgundy
            [148, 71, 102],   # Rose
            [55, 147, 19],    # Grass green
            [92, 201, 237],   # Baby blue
            [74, 217, 214],   # Aqua
            [209, 50, 169]    # Medium pink
        ]

    if dataset_name == 'coco':
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    elif dataset_name == 'NEU-DET':
        class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    elif dataset_name == 'UODD':
        class_names = ['seacucumber', 'seaurchin', 'scallop']
    elif dataset_name == 'ArTaxOr':
        class_names = ['Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera', 'Lepidoptera', 'Odonata']
    elif dataset_name == 'clipart1k':
        class_names = ['sheep', 'chair', 'boat', 'bottle', 'diningtable', 'sofa', 'cow', 'motorbike', 'car', 'aeroplane', 'cat', 'train', 'person', 'bicycle', 'pottedplant', 'bird', 'dog', 'bus', 'tvmonitor', 'horse']
    elif dataset_name == 'FISH':
        class_names = ['fish']
    elif dataset_name == 'DIOR':
        class_names = ['Expressway-Service-area', 'Expressway-toll-station', 'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill']
    elif dataset_name in METAINFO.keys():
        class_names = METAINFO[dataset_name]
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    img = Image.open(img_path)
    img_copy = copy.deepcopy(img)

    filter_inds = scores > score_thr
    scores = scores[filter_inds]
    bboxes = bboxes[filter_inds]
    labels = labels[filter_inds]

    class_inds = labels.tolist()
    scores = scores.tolist()
    if not show_scores:
        label_strs = [class_names[ind] for ind in class_inds]
    else:
        label_strs = [
            class_names[ind]+'=%d'%(s*100) for ind, s in zip(class_inds, scores)
        ]
    colors = [template_colors[ind] for ind in class_inds]

    gt_label_strs = [class_names[ind] for ind in gt_labels]
    gt_colors = [template_colors[ind] for ind in gt_labels]
    
    if len(labels) == 0 or len(bboxes) == 0 or len(scores) == 0:
        return

    try:
        draw_box_on_image(
            img,
            bboxes.tolist(),
            label_strs,
            show_label=True,
            colors=colors,
        )
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        print(f"Continue")
        return

    if len(gt_bboxes) > 0:
        try:
            draw_box_on_image(
                img_copy,
                gt_bboxes.tolist(),
                gt_label_strs,
                show_label=True,
                colors=gt_colors
            )
        except Exception as e:
            print(f"Error drawing GT boxes: {e}")
            print(f"Continue")
            return

    vis_np = np.array(img)
    gt_vis_np = np.array(img_copy)

    margin = np.zeros((vis_np.shape[0], 10, 3), dtype=vis_np.dtype) + 255
    vis_out = np.concatenate((gt_vis_np, margin, vis_np), axis=1)

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    Image.fromarray(vis_out).save(out_path)
