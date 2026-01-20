import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# --- CONFIG ---
SOURCE_DIRS = {
    "dinov2": "./results_analysis/few_shot_classes_dinov2_large",
    "dinov3_b": "./results_analysis/few_shot_classes_dinov3_base",
    "dinov3_l": "./results_analysis/few_shot_classes_dinov3_large",
    "dinov3_h": "./results_analysis/few_shot_classes_dinov3_huge",
    # "clipl": "./results_analysis/few_shot_classes_clip_l14",
    # "pel": "./results_analysis/few_shot_classes_PE-Spatial-L14-448",
}

SQUARE_IMAGES = True

DEST_DIR = Path("./results_analysis/paper_visualisations/dino_comparison")
# DEST_DIR = Path("./results_analysis/paper_visualisations/model_comparison")

IMAGE_IDS = [
    "000000000632",
    "000000015440",
    "000000015335",
    "000000014831",
    "000000013546",
    "000000013348",
    "000000012748",
    "000000009891",
    "000000009483",
]

# IMAGE_IDS = [
#     "000000000139",
#     "000000001268",
#     "000000001353",
#     "000000002153",
#     "000000002299",
#     "000000003255",
#     "000000004495",
#     "000000005193",
#     "000000005992",
#     "000000006723",
#     "000000007281",
#     "000000007511",
#     "000000007816",
#     "000000008021",
#     "000000009483",
#     "000000010363",
#     "000000012120",
#     "000000004134",
#     "000000007088",
#     "000000007574",
#     "000000011197",
# ]

# Ensure destination folder exists
DEST_DIR.mkdir(parents=True, exist_ok=True)

# --- SCRIPT ---
for model_name, src_root in SOURCE_DIRS.items():
    src_root = Path(src_root)

    for image_id in IMAGE_IDS:
        filename = image_id + ".jpg"
        src_file = src_root / filename

        if not src_file.exists():
            print(f"[WARN] Missing: {src_file}")
            continue

        # Load + convert to array
        img = Image.open(src_file).convert("RGB")
        arr = np.array(img)

        H, W, C = arr.shape

        # They were saved as:  [ GT | margin(10px) | prediction ]
        margin = 10
        w_left = (W - margin) // 2

        original = arr[:, :w_left, :]
        prediction = arr[:, w_left + margin :, :]

        # Convert to PIL Images
        original_img = Image.fromarray(original)
        prediction_img = Image.fromarray(prediction)
        
        # Optionally resize to square
        if SQUARE_IMAGES:
            # Use the maximum dimension
            max_dim = max(H, w_left)
            
            original_img = original_img.resize((max_dim, max_dim), Image.LANCZOS)
            prediction_img = prediction_img.resize((max_dim, max_dim), Image.LANCZOS)

        # Save left part as "_original"
        out_original = DEST_DIR / f"{image_id}_original.jpg"
        original_img.save(out_original)

        # Save right part as "_{model}.jpg"
        out_pred = DEST_DIR / f"{image_id}_{model_name}.jpg"
        prediction_img.save(out_pred)

        print(f"[INFO] Saved {out_original.name} and {out_pred.name}")

print("Done.")
