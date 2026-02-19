#!/usr/bin/env python3
"""
Compare BBox AP (IoU=0.50:0.95) between experiments without and with --heuristics.

Usage:
  python scripts/paper_figures/plot_EO_heuristic.py \
    --no-heuristics ./EO_results_no_heuristics \
    --heuristics ./EO_results \
    [--output ./EO_results]
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Styling (LaTeX-ready)
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# -----------------------------
# Constants
# -----------------------------
MODEL_ORDER = [
    "dinov2_l",
    "dinov3_l",
    "dinov3_sat_l",
    "DETIC",
    "DEViT",
]

# Subset of models for which we report a separate heuristic verdict (DINO-style only)
DINO_HEURISTIC_MODELS = {"dinov2_l", "dinov3_l", "dinov3_sat_l"}

SHOT_RE = re.compile(
    r"(?P<shot>\d+)_shot_(?P<model>dinov[23](?:_sat)?_l|DETIC|DEViT)_seed(?P<seed>\d+)"
)

ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


def parse_coco_eval(stats_path: Path):
    """
    Returns dict with at least "bbox_50_95" (and optionally bbox_50, segm_*),
    or None if parsing fails.
    """
    metrics = {}
    section = None
    found_bbox = False
    found_segm = False

    try:
        with stats_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("===== BBOX RESULTS"):
                    section = "bbox"
                    found_bbox = True
                elif s.startswith("===== SEGM RESULTS"):
                    section = "segm"
                    found_segm = True
                elif section and s.startswith("AP IoU=0.50:0.95:"):
                    try:
                        metrics[f"{section}_50_95"] = float(s.split(":")[-1])
                    except Exception:
                        pass
                elif section and s.startswith("AP IoU=0.50:"):
                    if not s.startswith("AP IoU=0.50:0.95:"):
                        try:
                            metrics[f"{section}_50"] = float(s.split(":")[-1])
                        except Exception:
                            pass
    except Exception:
        return None

    if found_bbox and not found_segm:
        required = {"bbox_50_95", "bbox_50"}
    else:
        required = {"bbox_50_95", "bbox_50", "segm_50_95", "segm_50"}
    if not required.issubset(metrics):
        return None

    return metrics


def load_root(root: Path):
    """
    Load from a base folder (no-heuristics or heuristics).

    Returns:
      dataset -> (shot, model) -> list of bbox_50_95 values (one per seed)
    """
    out = defaultdict(lambda: defaultdict(list))

    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name

        for exp_dir in dataset_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            m = SHOT_RE.fullmatch(exp_dir.name)
            if not m:
                continue

            shot = int(m.group("shot"))
            model = m.group("model")

            stats_path = exp_dir / "coco_eval_stats_.txt"
            if not stats_path.exists():
                continue

            metrics = parse_coco_eval(stats_path)
            if metrics is None:
                continue

            out[dataset][(shot, model)].append(metrics["bbox_50_95"])

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Plot BBox AP difference (heuristics vs no heuristics) per dataset."
    )
    parser.add_argument(
        "--no-heuristics",
        required=True,
        help="Base folder containing results WITHOUT --heuristics (dataset/model/shot structure).",
    )
    parser.add_argument(
        "--heuristics",
        required=True,
        help="Base folder containing results WITH --heuristics.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder for plots (default: same as --heuristics).",
    )
    args = parser.parse_args()

    no_h_root = Path(args.no_heuristics)
    h_root = Path(args.heuristics)
    output_root = Path(args.output) if args.output else h_root

    # Only create a subfolder "heuristics" INSIDE output_root.
    # If the user didn't specify --output (i.e. output_root == h_root), only create the heuristics subfolder.
    # Do NOT attempt to mkdir the existing output_root.

    output_dir = output_root / "heuristics"
    # Avoid mkdir(parents=True) when a parent is a symlink (see https://github.com/python/cpython/issues/83439)
    # Instead, make each part manually, skipping symlinks and existing directories
    current = output_root
    if not current.exists():
        raise RuntimeError(f"Output root does not exist: {current!s}")
    for part in output_dir.relative_to(output_root).parts:
        current = current / part
        if current.exists():
            continue  # Directory (or symlink) already exists
        try:
            current.mkdir()
        except FileExistsError:
            # This can happen if a component is a symlink; continue anyway
            continue

    if not no_h_root.is_dir():
        raise SystemExit(f"Not a directory: {no_h_root}")
    if not h_root.is_dir():
        raise SystemExit(f"Not a directory: {h_root}")

    data_no_h = load_root(no_h_root)
    data_h = load_root(h_root)

    all_datasets = sorted(set(data_no_h.keys()) | set(data_h.keys()))

    # Collect all (dataset, shot, model, delta) for global aggregate plots
    all_deltas = []

    for dataset in all_datasets:
        no_h_vals = data_no_h[dataset]  # (shot, model) -> list
        h_vals = data_h[dataset]

        # All (shot, model) keys that appear in either side
        all_keys = set(no_h_vals.keys()) | set(h_vals.keys())

        # Build deltas only where both sides have data; report missing
        deltas = []  # (shot, model, delta)
        missing_no_h = []
        missing_h = []

        for (shot, model) in sorted(all_keys):
            nl = no_h_vals.get((shot, model), [])
            hl = h_vals.get((shot, model), [])

            if not nl and not hl:
                continue
            if not nl:
                missing_no_h.append((dataset, shot, model))
                continue
            if not hl:
                missing_h.append((dataset, shot, model))
                continue

            mean_no_h = np.mean(nl)
            mean_h = np.mean(hl)
            delta = mean_h - mean_no_h
            deltas.append((shot, model, delta))
            all_deltas.append((dataset, shot, model, delta))

        # Print missing in orange
        for (ds, shot, model) in missing_no_h:
            print(f"{ORANGE}Missing (no_heuristics): dataset={ds} shot={shot} model={model}{RESET}")
        for (ds, shot, model) in missing_h:
            print(f"{ORANGE}Missing (heuristics): dataset={ds} shot={shot} model={model}{RESET}")

        if not deltas:
            print(f"Skipping dataset {dataset}: no (shot, model) with both results.")
            continue

        # Plot: grouped bars per shot, one bar per model; y = delta AP
        shots_sorted = sorted({s for s, m, d in deltas})
        models_in_data = sorted({m for s, m, d in deltas}, key=lambda x: (MODEL_ORDER.index(x) if x in MODEL_ORDER else 999, x))

        n_shots = len(shots_sorted)
        n_models = len(models_in_data)
        width = 0.8 / max(n_models, 1)
        x = np.arange(n_shots)

        fig, ax = plt.subplots(figsize=(max(6, n_shots * 1.2), 5))

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(models_in_data), 1)))
        model_to_color = {m: colors[i] for i, m in enumerate(models_in_data)}

        for i, model in enumerate(models_in_data):
            # Only plot bars where we have a delta (both sides present)
            x_indices = []
            heights = []
            for j, shot in enumerate(shots_sorted):
                d = next((delta for (s, m, delta) in deltas if s == shot and m == model), None)
                if d is not None:
                    x_indices.append(j)
                    heights.append(d)

            if not heights:
                continue

            x_indices = np.array(x_indices)
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(
                x_indices + offset,
                heights,
                width,
                label=model.replace("_", " "),
                color=model_to_color[model],
                edgecolor="gray",
                linewidth=0.5,
            )
            for bar, val in zip(bars, heights):
                bar.set_alpha(0.9 if val >= 0 else 0.65)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in shots_sorted])
        ax.set_xlabel("Number of shots")
        ax.set_ylabel(r"$\Delta$ BBox AP (heuristics $-$ no heuristics)")
        ax.set_title(f"{dataset}: effect of heuristics on BBox AP @ IoU=0.50:0.95")
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset}_heuristic_comparison.png", dpi=300)
        plt.close(fig)
        print(f"Saved {output_dir / f'{dataset}_heuristic_comparison.png'}")

    # -----------------------------
    # Global aggregate plots (saved in heuristics/global/)
    # -----------------------------
    global_dir = output_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    if all_deltas:
        # 1) Aggregate over datasets: overall per shot per model
        shot_model_to_deltas = defaultdict(list)
        for (ds, shot, model, delta) in all_deltas:
            shot_model_to_deltas[(shot, model)].append(delta)
        shot_model_means = {}
        for (shot, model), vals in shot_model_to_deltas.items():
            shot_model_means[(shot, model)] = np.mean(vals)

        shots_global = sorted({s for (s, m), v in shot_model_means.items()})
        models_global = sorted({m for (s, m), v in shot_model_means.items()},
                               key=lambda x: (MODEL_ORDER.index(x) if x in MODEL_ORDER else 999, x))
        n_s = len(shots_global)
        n_m = len(models_global)
        w = 0.8 / max(n_m, 1)
        fig1, ax1 = plt.subplots(figsize=(max(6, n_s * 1.2), 5))
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_m, 1)))
        for i, model in enumerate(models_global):
            x_idx, h = [], []
            for j, shot in enumerate(shots_global):
                key = (shot, model)
                if key in shot_model_means:
                    x_idx.append(j)
                    h.append(shot_model_means[key])
            if x_idx:
                ax1.bar(np.array(x_idx) + (i - (n_m - 1) / 2) * w, h, w, label=model.replace("_", " "),
                        color=colors[i], edgecolor="gray", linewidth=0.5)
        ax1.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax1.set_xticks(np.arange(n_s))
        ax1.set_xticklabels([str(s) for s in shots_global])
        ax1.set_xlabel("Number of shots")
        ax1.set_ylabel(r"$\Delta$ BBox AP (mean over datasets)")
        ax1.set_title("Global: effect of heuristics per shot per model (aggregated over datasets)")
        ax1.legend(loc="best")
        ax1.grid(True, axis="y", alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(global_dir / "global_per_shot_per_model.png", dpi=300)
        plt.close(fig1)
        print(f"Saved {global_dir / 'global_per_shot_per_model.png'}")

        # 2) Aggregate over shots: per dataset per model
        ds_model_to_deltas = defaultdict(list)
        for (ds, shot, model, delta) in all_deltas:
            ds_model_to_deltas[(ds, model)].append(delta)
        ds_model_means = {k: np.mean(v) for k, v in ds_model_to_deltas.items()}
        datasets_global = sorted({ds for (ds, m) in ds_model_means})
        models_dm = sorted({m for (ds, m) in ds_model_means},
                          key=lambda x: (MODEL_ORDER.index(x) if x in MODEL_ORDER else 999, x))
        n_ds = len(datasets_global)
        n_m2 = len(models_dm)
        w2 = 0.8 / max(n_m2, 1)
        fig2, ax2 = plt.subplots(figsize=(max(8, n_ds * 0.7), 5))
        for i, model in enumerate(models_dm):
            vals = [ds_model_means.get((ds, model), np.nan) for ds in datasets_global]
            # use nan for missing so bar is skipped
            x_idx = [j for j in range(n_ds) if not np.isnan(vals[j])]
            v = [vals[j] for j in x_idx]
            if x_idx:
                ax2.bar(np.array(x_idx) + (i - (n_m2 - 1) / 2) * w2, v, w2, label=model.replace("_", " "),
                        color=plt.cm.tab10(i % 10), edgecolor="gray", linewidth=0.5)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax2.set_xticks(np.arange(n_ds))
        ax2.set_xticklabels(datasets_global, rotation=45, ha="right")
        ax2.set_xlabel("Dataset")
        ax2.set_ylabel(r"$\Delta$ BBox AP (mean over shots)")
        ax2.set_title("Per dataset per model: effect of heuristics (aggregated over shots)")
        ax2.legend(loc="best")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(global_dir / "global_per_dataset_per_model.png", dpi=300)
        plt.close(fig2)
        print(f"Saved {global_dir / 'global_per_dataset_per_model.png'}")

        # 3) Aggregate over models: per dataset per shot
        ds_shot_to_deltas = defaultdict(list)
        for (ds, shot, model, delta) in all_deltas:
            ds_shot_to_deltas[(ds, shot)].append(delta)
        ds_shot_means = {k: np.mean(v) for k, v in ds_shot_to_deltas.items()}
        shots_ds = sorted({s for (ds, s) in ds_shot_means})
        n_sh = len(shots_ds)
        w3 = 0.8 / max(n_sh, 1)
        fig3, ax3 = plt.subplots(figsize=(max(8, n_ds * 0.7), 5))
        for i, shot in enumerate(shots_ds):
            vals = [ds_shot_means.get((ds, shot), np.nan) for ds in datasets_global]
            x_idx = [j for j in range(n_ds) if not np.isnan(vals[j])]
            v = [vals[j] for j in x_idx]
            if x_idx:
                ax3.bar(np.array(x_idx) + (i - (n_sh - 1) / 2) * w3, v, w3, label=f"{shot}-shot",
                        color=plt.cm.tab10(i % 10), edgecolor="gray", linewidth=0.5)
        ax3.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax3.set_xticks(np.arange(n_ds))
        ax3.set_xticklabels(datasets_global, rotation=45, ha="right")
        ax3.set_xlabel("Dataset")
        ax3.set_ylabel(r"$\Delta$ BBox AP (mean over models)")
        ax3.set_title("Per dataset per shot: effect of heuristics (aggregated over models)")
        ax3.legend(loc="best")
        ax3.grid(True, axis="y", alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(global_dir / "global_per_dataset_per_shot.png", dpi=300)
        plt.close(fig3)
        print(f"Saved {global_dir / 'global_per_dataset_per_shot.png'}")

        # 4) Aggregate over shots and models: overall per dataset
        ds_to_deltas = defaultdict(list)
        for (ds, shot, model, delta) in all_deltas:
            ds_to_deltas[ds].append(delta)
        ds_means = {ds: np.mean(v) for ds, v in ds_to_deltas.items()}
        datasets_sorted = sorted(ds_means.keys())
        means_list = [ds_means[ds] for ds in datasets_sorted]
        fig4, ax4 = plt.subplots(figsize=(max(8, len(datasets_sorted) * 0.6), 5))
        colors4 = ["#2ecc71" if m >= 0 else "#e74c3c" for m in means_list]
        ax4.bar(np.arange(len(datasets_sorted)), means_list, color=colors4, edgecolor="gray", linewidth=0.5)
        ax4.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax4.set_xticks(np.arange(len(datasets_sorted)))
        ax4.set_xticklabels(datasets_sorted, rotation=45, ha="right")
        ax4.set_xlabel("Dataset")
        ax4.set_ylabel(r"$\Delta$ BBox AP (mean over shots and models)")
        ax4.set_title("Overall per dataset: effect of heuristics (aggregated over shots and models)")
        ax4.grid(True, axis="y", alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(global_dir / "global_per_dataset.png", dpi=300)
        plt.close(fig4)
        print(f"Saved {global_dir / 'global_per_dataset.png'}")

    # -----------------------------
    # Overall verdict: did heuristics add value?
    # -----------------------------
    if all_deltas:
        all_delta_values = [d for (_, _, _, d) in all_deltas]
        mean_delta = np.mean(all_delta_values)
        n_points = len(all_delta_values)
        if mean_delta > 0:
            verdict = "added value"
            msg = (
                f"\nOverall (all datasets, shots, models): heuristics {verdict}. "
                f"Mean Δ BBox AP = +{mean_delta:.4f} (n={n_points} experiment settings)."
            )
        elif mean_delta < 0:
            verdict = "reduced performance on average"
            msg = (
                f"\nOverall (all datasets, shots, models): heuristics {verdict}. "
                f"Mean Δ BBox AP = {mean_delta:.4f} (n={n_points} experiment settings)."
            )
        else:
            msg = (
                f"\nOverall (all datasets, shots, models): no average change. "
                f"Mean Δ BBox AP = 0.0000 (n={n_points} experiment settings)."
            )
        print(msg)

        # Same verdict but only for dinov2, dinov3, dinov3_sat
        dino_deltas = [
            (ds, shot, model, d)
            for (ds, shot, model, d) in all_deltas
            if model.lower() in {m.lower() for m in DINO_HEURISTIC_MODELS}
        ]
        if dino_deltas:
            dino_delta_values = [d for (_, _, _, d) in dino_deltas]
            mean_dino = np.mean(dino_delta_values)
            n_dino = len(dino_delta_values)
            if mean_dino > 0:
                msg_dino = (
                    f"Overall (dinov2 / dinov3 / dinov3_sat only): heuristics added value. "
                    f"Mean Δ BBox AP = +{mean_dino:.4f} (n={n_dino} experiment settings)."
                )
            elif mean_dino < 0:
                msg_dino = (
                    f"Overall (dinov2 / dinov3 / dinov3_sat only): heuristics reduced performance on average. "
                    f"Mean Δ BBox AP = {mean_dino:.4f} (n={n_dino} experiment settings)."
                )
            else:
                msg_dino = (
                    f"Overall (dinov2 / dinov3 / dinov3_sat only): no average change. "
                    f"Mean Δ BBox AP = 0.0000 (n={n_dino} experiment settings)."
                )
            print(msg_dino)
        else:
            print("Overall (dinov2 / dinov3 / dinov3_sat only): no comparable pairs — cannot compute verdict.")
    else:
        print("\nOverall: no comparable (dataset, shot, model) pairs — cannot compute verdict.")

    print(f"\nPer-dataset plots saved to {output_dir}")
    print(f"Global aggregate plots saved to {global_dir}")


if __name__ == "__main__":
    main()
