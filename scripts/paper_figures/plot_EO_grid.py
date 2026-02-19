#!/usr/bin/env python3
"""
Build grid visualisations for the camera-ready paper: for a given dataset and shot count,
gather PNGs from each model's results_analysis (GT|pred layout), extract the prediction
half, keep one ground truth, and horizontally stack: GT, dinov2, dinov3, dinov3-sat, devit, detic.

Usage:
  python scripts/paper_figures/plot_EO_grid.py \
    --root ./EO_results \
    --dataset ISAID \
    --shots 1 2 3 5 10
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Constants (aligned with plot_EO_accuracy.py)
# -----------------------------
SHOT_RE = re.compile(
    r"(?P<shot>\d+)_shot_(?P<model>dinov[23](?:_sat)?_l|DETIC|DEViT)_seed(?P<seed>\d+)"
)

# Output column order: ground truth, then models in this order
MODEL_ORDER = [
    "dinov2_l",
    "dinov3_l",
    "dinov3_sat_l",
    "DEViT",
    "DETIC",
]

# Known EO dataset names (case-insensitive); used for validation
VALID_DATASETS = {
    "FAST", "HRSID", "ISAID", "MAPPING", "NWPU", "SIOR", "SODAA", "SOTA",
    "SSDD", "VEDAI512", "VEDAI1024", "XVIEW", "RAREPLANES", "RAREPLANES_SINGLE_CLASS",
}

VALID_SHOTS = {1, 2, 3, 5, 10}

# Default margin between left (GT) and right (pred) in source PNGs (see visualization.py).
# We skip this strip when splitting, so the saved grids are correct.
DEFAULT_MARGIN = 10
# White gap (pixels) between concatenated panels in the output grid (0 = no gap).
DEFAULT_GAP = 5

# Some models use "visualizations" instead of "results_analysis"; we try preferred first, then fallback
MODEL_VIS_FOLDER = {"DEViT": "visualizations", "DETIC": "visualizations"}
DEFAULT_VIS_FOLDER = "results_analysis"


def _resolve_dataset_name(root: Path, dataset: str) -> Path:
    """Resolve dataset to a directory under root. Validates spelling (case-insensitive)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Root is not a directory: {root}")

    # Allow user to pass exact folder name (e.g. ISAID) or we match case-insensitively
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name == dataset or d.name.upper() == dataset.upper():
            return d

    # Suggest valid names from registry if we have no match
    available = sorted({d.name for d in root.iterdir() if d.is_dir()})
    valid_str = ", ".join(sorted(VALID_DATASETS))
    raise ValueError(
        f"Dataset '{dataset}' not found under {root}. "
        f"Available dirs: {available}. "
        f"Valid dataset names (EO): {valid_str}"
    )


def _normalize_image_key(rel_path: str) -> str:
    """
    Normalize an image path key by stripping trailing .png and .jpg (repeatedly).
    E.g. asdf.png, asdf.jpg, asdf.png.jpg, asdf.jpg.png all become the same key.
    Preserves directory part (e.g. 'subdir/asdf.png' -> 'subdir/asdf').
    """
    path = Path(rel_path)
    parent = path.parent
    name = path.name
    # Strip .png and .jpg from the end until no more
    while True:
        lower = name.lower()
        if lower.endswith(".png"):
            name = name[:-4]
        elif lower.endswith(".jpg"):
            name = name[:-4]
        else:
            break
    if parent and str(parent) != ".":
        return (parent / name).as_posix()
    return name


def _collect_png_paths(vis_dir: Path, use_basename_only: bool = True) -> dict[str, Path]:
    """
    Collect image paths under vis_dir (.png and .jpg), including subdirs.
    Returns mapping: match key -> absolute Path.
    If use_basename_only, match key is the normalized filename only (no subdir), so that
    e.g. dinov2_large/P0013_xxx.jpg and dinov3_large/P0013_xxx.jpg and P0013_xxx.jpg all match as P0013_xxx.
    Same logical image (asdf.png vs asdf.jpg) yields the same key; we keep one path per key (prefer .png).
    """
    out = {}
    if not vis_dir.is_dir():
        return out
    for ext in ("*.png", "*.jpg"):
        for p in vis_dir.rglob(ext):
            try:
                rel = p.relative_to(vis_dir)
                key = _normalize_image_key(rel.as_posix())
                if use_basename_only:
                    key = Path(key).name
                if key not in out or p.suffix.lower() == ".png":
                    out[key] = p
            except ValueError:
                continue
            except Exception:
                continue
    return out


def _split_gt_pred(image: np.ndarray, margin: int = DEFAULT_MARGIN):
    """
    Split image laid out as [ GT | margin | prediction ] (axis=1).
    Returns (gt_array, pred_array) with same height; each side has width (W - margin) // 2.
    """
    h, w = image.shape[:2]
    half = (w - margin) // 2
    gt = image[:, :half].copy()
    pred = image[:, half + margin :].copy()
    return gt, pred


def _pick_one_exp_per_model(dataset_dir: Path, shot: int, seed_preference: int | None) -> dict[str, Path]:
    """
    For the given dataset dir and shot count, find one experiment dir per model.
    Returns dict: model_name -> path to experiment dir.
    """
    by_model = defaultdict(list)
    for exp_dir in dataset_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        m = SHOT_RE.fullmatch(exp_dir.name)
        if not m or int(m.group("shot")) != shot:
            continue
        model = m.group("model")
        seed = int(m.group("seed"))
        by_model[model].append((seed, exp_dir))

    result = {}
    for model in MODEL_ORDER:
        candidates = by_model.get(model)
        if not candidates:
            continue
        # Prefer given seed if present, else first by seed order
        if seed_preference is not None:
            for s, p in sorted(candidates, key=lambda x: (x[0] != seed_preference, x[0])):
                result[model] = p
                break
        else:
            result[model] = min(candidates, key=lambda x: x[0])[1]
    return result


def run_grid(root: Path, dataset_name: str, shots: list[int], seed: int | None, margin: int, gap: int):
    root = Path(root)
    dataset_dir = _resolve_dataset_name(root, dataset_name)
    # Use dataset_dir.name for output so folder name matches filesystem
    dataset_label = dataset_dir.name

    for shot in shots:
        if shot not in VALID_SHOTS:
            print(f"Skipping invalid shot count: {shot} (allowed: {sorted(VALID_SHOTS)})")
            continue
        model_to_exp = _pick_one_exp_per_model(dataset_dir, shot, seed)
        if len(model_to_exp) < len(MODEL_ORDER):
            missing = set(MODEL_ORDER) - set(model_to_exp.keys())
            print(f"[{dataset_label}] {shot}-shot: missing models {missing}, skipping.")
            continue

        # Collect image paths per model (try model-specific folder, then results_analysis if empty)
        model_to_pngs: dict[str, dict[str, Path]] = {}
        model_to_vis_dir: dict[str, Path] = {}
        for model, exp_path in model_to_exp.items():
            folder = MODEL_VIS_FOLDER.get(model, DEFAULT_VIS_FOLDER)
            vis_dir = exp_path / folder
            paths = _collect_png_paths(vis_dir, use_basename_only=True)
            if not paths and folder != DEFAULT_VIS_FOLDER:
                vis_dir = exp_path / DEFAULT_VIS_FOLDER
                paths = _collect_png_paths(vis_dir, use_basename_only=True)
            model_to_pngs[model] = paths
            model_to_vis_dir[model] = vis_dir
            
        # Debug: Print per-model image counts and paths
        print(f"\n[DEBUG] {dataset_label} {shot}-shot image collection summary:")
        all_image_ids = set()
        for model in MODEL_ORDER:
            if model in model_to_pngs:
                all_image_ids |= set(model_to_pngs[model].keys())
        total_unique = len(all_image_ids)
        print(f"  Total unique image IDs across all models: {total_unique}")
        for model in MODEL_ORDER:
            if model in model_to_pngs:
                count = len(model_to_pngs[model])
                missing_count = total_unique - count
                coverage = 100 * count / total_unique if total_unique > 0 else 0
                print(f"  {model}: {count} images (missing {missing_count}, coverage {coverage:.1f}%) — {model_to_vis_dir[model]}")

        # Debug: show sample keys per model to spot naming mismatches
        print(f"  Sample keys per model (first 5):")
        for model in MODEL_ORDER:
            if model in model_to_pngs:
                sample = sorted(model_to_pngs[model].keys())[:5]
                print(f"    {model}: {sample}")

        # Debug: pairwise intersection sizes
        print(f"  Pairwise intersections:")
        models_present = [m for m in MODEL_ORDER if m in model_to_pngs]
        for i, m1 in enumerate(models_present):
            for m2 in models_present[i+1:]:
                shared = set(model_to_pngs[m1].keys()) & set(model_to_pngs[m2].keys())
                print(f"    {m1} ∩ {m2}: {len(shared)}")

        # Debug: progressive intersection — find which model causes the drop
        print(f"  Progressive intersection (adding models one by one):")
        running = set(model_to_pngs[models_present[0]].keys())
        print(f"    After {models_present[0]}: {len(running)}")
        for model in models_present[1:]:
            running &= set(model_to_pngs[model].keys())
            print(f"    After adding {model}: {len(running)}")

        # Only keep image basenames that appear in every model
        all_keys = set(model_to_pngs[MODEL_ORDER[0]].keys())
        for model in MODEL_ORDER[1:]:
            all_keys &= set(model_to_pngs[model].keys())
        print(f"  Common images across ALL models: {len(all_keys)}")

        if not all_keys:
            print(f"[{dataset_label}] {shot}-shot: no common PNGs across all models.")
            # Identify which models are missing images for each key
            keys_union = set()
            for model in MODEL_ORDER:
                keys_union |= set(model_to_pngs[model].keys())
            for k in sorted(keys_union):
                missing = [model for model in MODEL_ORDER if k not in model_to_pngs[model]]
                if missing:
                    print(f"  Image '{k}' missing from models: {', '.join(missing)}")
            continue


        out_dir = root / dataset_label / f"{shot}_shot_merged"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use first model (e.g. dinov2_l) to get ground truth once per image
        gt_model = MODEL_ORDER[0]
        n_saved = 0
        for key in tqdm(sorted(all_keys), desc=f"{dataset_label} {shot}-shot", unit="img"):
            try:
                img_path = model_to_pngs[gt_model][key]
                arr = np.array(Image.open(img_path).convert("RGB"))
                gt_img, _ = _split_gt_pred(arr, margin)
            except Exception as e:
                print(f"  Skip {key}: failed to load/split GT — {e}")
                continue

            parts = [gt_img]
            for model in MODEL_ORDER:
                p = model_to_pngs[model][key]
                arr = np.array(Image.open(p).convert("RGB"))
                _, pred = _split_gt_pred(arr, margin)
                # Resize pred to GT height if needed so stacking is clean
                if pred.shape[0] != gt_img.shape[0]:
                    pred = np.array(
                        Image.fromarray(pred).resize(
                            (gt_img.shape[1], gt_img.shape[0]),
                            Image.Resampling.LANCZOS,
                        )
                    )
                parts.append(pred)

            # Optionally insert white gaps between panels for readability
            if gap > 0:
                h = parts[0].shape[0]
                white = np.full((h, gap, 3), 255, dtype=parts[0].dtype)
                with_gaps = [parts[0]]
                for p in parts[1:]:
                    with_gaps.append(white)
                    with_gaps.append(p)
                merged = np.concatenate(with_gaps, axis=1)
            else:
                merged = np.concatenate(parts, axis=1)
            out_name = f"{key}.png"
            out_path = out_dir / out_name
            Image.fromarray(merged).save(out_path)
            n_saved += 1

        print(f"[{dataset_label}] {shot}-shot: saved {n_saved} grids to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build grid visualisations (GT + all model predictions) for paper."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root folder containing dataset result dirs (e.g. EO_results).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. ISAID, xview). Must match a folder under --root.",
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[1, 2, 3, 5, 10],
        help="Shot counts to process (default: 1 2 3 5 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Prefer this seed when multiple runs exist per model (default: 42).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=DEFAULT_MARGIN,
        help="Pixel margin between GT and prediction in source PNGs (default: 10).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=DEFAULT_GAP,
        help="White gap in pixels between concatenated panels in the output (default: 5). Use 0 for no gap.",
    )
    args = parser.parse_args()

    run_grid(
        root=Path(args.root),
        dataset_name=args.dataset,
        shots=args.shots,
        seed=args.seed,
        margin=args.margin,
        gap=args.gap,
    )


if __name__ == "__main__":
    main()
