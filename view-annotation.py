import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def visualize_coco_annotation(annotation_id, coco_json_path, output_path):
    # Initialize COCO api for instance annotations
    coco = COCO(coco_json_path)

    # Load annotation
    # ann_ids = coco.getAnnIds(ids=[annotation_id])
    ann_ids = [annotation_id]
    if not ann_ids:
        print(f"Annotation with ID {annotation_id} not found.")
        return

    ann = coco.loadAnns(ann_ids)[0]

    # Load image
    img_id = ann['image_id']
    img_info = coco.loadImgs(img_id)[0]
    
    # Check if the image is local or needs to be downloaded
    if 'coco_url' in img_info:
        response = requests.get(img_info['coco_url'])
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_info['file_name'])

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Display the image
    ax.imshow(img)

    # Draw the annotation
    mask = coco.annToMask(ann)
    masked = np.ma.masked_where(mask == 0, mask)
    print("masked.shape: ", masked.shape)
    print("masked.sum(): ", masked.sum())
    print("mask.sum(): ", mask.sum())
    print("annotation supposed area: ", ann['area'])
    print("annotation bbox: ", ann['bbox'])
    print("annotation", ann)
    ax.imshow(masked, alpha=0.5, cmap='jet')

    # Set title
    category_id = ann['category_id']
    category_name = coco.loadCats(category_id)[0]['name']
    ax.set_title(f"Annotation ID: {annotation_id}, Category: {category_name}")

    # Remove axes
    ax.axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    plt.close(fig)

if __name__ == "__main__":
    annotation_id = 1864550  # Replace with the desired annotation ID
    split = "train2017"
    annotation_json_name = f"instances_{split}_tiny_filtered_by_0.6"
    coco_json_path = f"/localdisk/data2/Users/s2254242/datasets/coco/annotations/{annotation_json_name}.json"
    output_path = "view-annotation.png"  # Replace with desired output path

    visualize_coco_annotation(annotation_id, coco_json_path, output_path)
