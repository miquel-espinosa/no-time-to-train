#!/usr/bin/env python3
"""
Visualise the PCA of DINOv2 patch features for a given input image.
Also saves a center-cropped, squared version of the input image alongside the PCA.
Usage:
    python scripts/paper_figures/plot_dinov2_pca.py --image path/to/image.jpg
    python scripts/paper_figures/plot_dinov2_pca.py --image path/to/image.jpg --output pca_vis.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.transforms import Normalize

import dinov2.dinov2.models.vision_transformer as dinov2_vit
import dinov2.dinov2.utils.utils as dinov2_utils
from no_time_to_train.models.matching_baseline_utils import vis_pca

CHECKPOINT_PATH = "checkpoints/dinov2/dinov2_vitl14_pretrain.pth"
IMG_SIZE = 518
PATCH_SIZE = 14
FEAT_DIM = 1024


def load_dinov2(checkpoint_path, device):
    encoder = dinov2_vit.vit_large(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        init_values=1e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )
    dinov2_utils.load_pretrained_weights(encoder, checkpoint_path, "teacher")
    return encoder.eval().to(device)


def extract_features(encoder, img_normalized, device):
    """Extract patch features the same way as Sam2MatchingBaseline."""
    with torch.no_grad():
        x = encoder.prepare_tokens_with_masks(img_normalized)
        n_skip_tokens = 1 + encoder.num_register_tokens
        for blk in encoder.blocks:
            x = blk(x)
        x = encoder.norm(x)
        feats = x[:, n_skip_tokens:]
    return feats.reshape(-1, FEAT_DIM)


def main():
    parser = argparse.ArgumentParser(description="DINOv2 PCA feature visualisation")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save the PCA output figure")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--transparency", type=float, default=1,
                        help="Blending weight for PCA overlay (0=original, 1=PCA only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = load_dinov2(args.checkpoint, device)

    encoder_hw = IMG_SIZE // PATCH_SIZE

    # Load and preprocess image
    img = Image.open(args.image).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Also save the center cropped (squared) image for reference.
    # We'll use the same transform but up to before ToTensor().
    crop_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
    ])
    img_cropped = crop_transform(img)  # PIL Image

    # Compute output paths
    image_path = Path(args.image)
    out_dir = image_path.parent
    stem = image_path.stem  # Without suffix

    # Determine default output paths if not specified
    # Default: foo.jpg -> foo_pca.png and foo_squared.jpg in the same folder as input
    if args.output is not None:
        output_path_pca = args.output
    else:
        output_path_pca = out_dir / f"{stem}_pca.png"
    output_path_sq = out_dir / f"{stem}_squared.jpg"

    img_normalized = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(img_tensor)

    print("Extracting features...")
    feats = extract_features(encoder, img_normalized, device)

    # Compute PCA (3 components -> RGB)
    pca = PCA(n_components=3)
    pca.fit(feats.cpu().numpy())
    pca_mean = torch.from_numpy(pca.mean_).to(device=device, dtype=feats.dtype)
    pca_components = torch.from_numpy(pca.components_).to(device=device, dtype=feats.dtype)

    # Wrap into the format expected by vis_pca (single-class tensors indexed by cat_ind=0)
    pca_means = pca_mean.unsqueeze(0)            # [1, dim]
    pca_comps = pca_components.unsqueeze(0)       # [1, 3, dim]

    # Use all patches as foreground
    ref_masks = torch.ones(encoder_hw * encoder_hw, device=device)
    ref_masks_ori = torch.ones(1, 1, IMG_SIZE, IMG_SIZE, device=device)

    encoder_shape_info = dict(height=encoder_hw, width=encoder_hw, patch_size=PATCH_SIZE)

    pca_vis = vis_pca(
        ref_imgs=img_tensor,
        ref_masks_ori=ref_masks_ori,
        ref_cat_ind=0,
        ref_feats=feats,
        ref_masks=ref_masks,
        pca_means=pca_means,
        pca_components=pca_comps,
        encoder_shape_info=encoder_shape_info,
        device=device,
        transparency=args.transparency,
    )

    result = pca_vis.clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Save PCA visualisation
    plt.figure(figsize=(8, 8))
    plt.imshow(result)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path_pca, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved PCA visualisation to {output_path_pca}")

    # Save center cropped squared image
    img_cropped.save(output_path_sq)
    print(f"Saved squared (center cropped) image to {output_path_sq}")


if __name__ == "__main__":
    main()
