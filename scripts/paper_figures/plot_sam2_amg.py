#!/usr/bin/env python3
"""
Visualise SAM2 Automatic Mask Generation (AMG) on a center-cropped input image.
Also saves a center-cropped, squared version of the input image (consistent with PCA DINOv2 feature script).

Usage:
    python scripts/paper_figures/plot_sam2_amg.py --image example_images/giraffe.jpg
    python scripts/paper_figures/plot_sam2_amg.py --image example_images/giraffe.jpg --output example_images
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

SAM2_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
SAM2_CONFIG = "sam2_hiera_l.yaml"
SAM2_INPUT_SIZE = 1036  # To match with PCA/DINOv2 and center cropping

def show_anns(masks, colors, borders=True):
    """Official SAM2/SAV visualisation of mask annotations."""
    if len(masks) == 0:
        return

    sorted_annot_and_color = sorted(
        zip(masks, colors), key=(lambda x: x[0].sum()), reverse=True
    )
    H, W = sorted_annot_and_color[0][0].shape[0], sorted_annot_and_color[0][0].shape[1]

    canvas = np.ones((H, W, 4))
    canvas[:, :, 3] = 0
    contour_thickness = max(1, int(min(5, 0.01 * min(H, W))))
    for mask, color in sorted_annot_and_color:
        canvas[mask] = np.concatenate([color, [0.55]])
        if borders:
            contours, _ = cv2.findContours(
                np.array(mask, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(
                canvas, contours, -1, (0.05, 0.05, 0.05, 1), thickness=contour_thickness
            )

    ax = plt.gca()
    ax.imshow(canvas)


def main():
    parser = argparse.ArgumentParser(description="SAM2 automatic mask generation (center cropped input)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (file or directory); default: image folder with default names")
    parser.add_argument("--checkpoint", type=str, default=SAM2_CHECKPOINT)
    parser.add_argument("--config", type=str, default=SAM2_CONFIG)
    parser.add_argument("--points-per-side", type=int, default=32,
                        help="Grid density for point prompts (default 32 in SAM2, 64 here for more masks)")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7,
                        help="Predicted IoU threshold to keep a mask (lower = more masks, default 0.8 in SAM2)")
    parser.add_argument("--stability-score-thresh", type=float, default=0.9,
                        help="Stability score threshold (lower = more masks, default 0.95 in SAM2)")
    args = parser.parse_args()

    # Output and naming setup
    image_path = Path(args.image)
    out_dir = Path(args.output) if args.output else image_path.parent
    stem = image_path.stem
    output_path_sam = out_dir / f"{stem}_sam2_amg.png"
    output_path_sq = out_dir / f"{stem}_squared.jpg"

    # Load and center-crop image (same as plot_dinov2_pca.py)
    pil_img = Image.open(args.image).convert("RGB")
    transform_sq = transforms.Compose([
        transforms.Resize(SAM2_INPUT_SIZE),
        transforms.CenterCrop(SAM2_INPUT_SIZE),
    ])
    img_cropped = transform_sq(pil_img)
    img_sam = np.array(img_cropped)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_cropped.save(output_path_sq)
    print(f"Saved squared (center cropped) image to {output_path_sq}")

    print(f"Loading SAM2 from {args.checkpoint}...")
    sam2 = build_sam2(args.config, args.checkpoint, device="cuda")
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )

    print("Generating masks (on center-cropped area)...")
    amg_results = mask_generator.generate(img_sam)
    print(f"Found {len(amg_results)} masks")

    mask_arrays = [r["segmentation"] for r in amg_results]
    colors = np.random.random((len(mask_arrays), 3))

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_sam)
    show_anns(mask_arrays, colors)
    ax.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(output_path_sam, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved SAM2 AMG visualisation to {output_path_sam}")


if __name__ == "__main__":
    main()
