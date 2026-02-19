#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

"""
Usage:
python scripts/paper_figures/plot_EO_accuracy.py \
  --input-root ./EO_results \
  --output-root ./EO_results
"""

# -----------------------------
# Styling (LaTeX-ready)
# -----------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
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
# New SHOT_RE pattern to accept dinov3_l, dinov2_l, dinov3_sat_l, DETIC, DEViT
SHOT_RE = re.compile(
    r"(?P<shot>\d+)_shot_(?P<model>dinov[23](?:_sat)?_l|DETIC|DEViT)_seed(?P<seed>\d+)"
)
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"

# -----------------------------
# Parsing helpers
# -----------------------------

def parse_coco_eval(stats_path: Path):
    """
    Returns:
      {
        "bbox_50_95": float,
        "bbox_50": float,
        "segm_50_95": float,
        "segm_50": float,
      }
    or None if parsing fails.
    Accepts files containing only BBox results.
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
                    # avoid double-matching the 0.50:0.95 line
                    if not s.startswith("AP IoU=0.50:0.95:"):
                        try:
                            metrics[f"{section}_50"] = float(s.split(":")[-1])
                        except Exception:
                            pass
    except Exception:
        return None

    # If only bbox section exists, require only those keys
    if found_bbox and not found_segm:
        required = {"bbox_50_95", "bbox_50"}
    # Normal case, require both
    else:
        required = {"bbox_50_95", "bbox_50", "segm_50_95", "segm_50"}
    if not required.issubset(metrics):
        return None

    return metrics


# -----------------------------
# Summary plot helpers (use bbox_50_95 as primary metric)
# -----------------------------
ACCURACY_KEY = "bbox_50_95"


def _mean_ap(data_list):
    """Mean of bbox_50_95 over a list of metric dicts."""
    vals = [d.get(ACCURACY_KEY) for d in data_list if d.get(ACCURACY_KEY) is not None]
    return np.mean(vals) if vals else np.nan


def _all_shots(global_data):
    shots = set()
    for d_data in global_data.values():
        for m_data in d_data.values():
            shots.update(m_data.keys())
    return sorted(shots)


def plot_summary_aggregated_vs_shots(global_data, output_root):
    """Aggregated accuracy across datasets and models vs number of shots (one line)."""
    all_shot_vals = _all_shots(global_data)
    if not all_shot_vals:
        return
    means, stds = [], []
    for shot in all_shot_vals:
        vals = []
        for d_data in global_data.values():
            for m_data in d_data.values():
                if shot in m_data:
                    v = _mean_ap(m_data[shot])
                    if not np.isnan(v):
                        vals.append(v)
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if len(vals) > 1 else 0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(all_shot_vals, means, yerr=stds, marker="o", capsize=3)
    ax.set_xlabel("Number of shots")
    ax.set_ylabel("BBox AP @ IoU=0.50:0.95")
    ax.set_title("Aggregated accuracy (all datasets, all models)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_root / "summary_aggregated_vs_shots.png", dpi=300)
    plt.close(fig)


def plot_summary_overall_by_model(global_data, output_root):
    """Overall accuracy per model (bar), aggregated across shots and datasets."""
    model_scores = []
    for model in MODEL_ORDER:
        vals = []
        for d_data in global_data.values():
            m_key = next((k for k in d_data if k.lower() == model.lower()), None)
            if m_key is None:
                continue
            for shot_list in d_data[m_key].values():
                v = _mean_ap(shot_list)
                if not np.isnan(v):
                    vals.append(v)
        if vals:
            model_scores.append((model.replace("_", " "), np.mean(vals)))
    if not model_scores:
        return
    names, scores = zip(*model_scores)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, scores, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("BBox AP @ IoU=0.50:0.95")
    ax.set_title("Overall accuracy by model (aggregated across shots and datasets)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_root / "summary_overall_by_model.png", dpi=300)
    plt.close(fig)


def plot_summary_overall_by_dataset(global_data, output_root):
    """Overall accuracy per dataset (bar), aggregated across models and shots."""
    dataset_scores = []
    for dataset in sorted(global_data.keys()):
        vals = []
        for m_data in global_data[dataset].values():
            for shot_list in m_data.values():
                v = _mean_ap(shot_list)
                if not np.isnan(v):
                    vals.append(v)
        dataset_scores.append((dataset, np.mean(vals) if vals else np.nan))
    if not dataset_scores:
        return
    names, scores = zip(*dataset_scores)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 4))
    ax.bar(x, scores, color="seagreen", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("BBox AP @ IoU=0.50:0.95")
    ax.set_title("Overall accuracy per dataset (aggregated across models and shots)")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_root / "summary_overall_by_dataset.png", dpi=300)
    plt.close(fig)


def plot_summary_model_per_dataset(global_data, output_root):
    """Accuracy per model per dataset (bar), aggregated across shots."""
    datasets = sorted(global_data.keys())
    model_keys_in_data = set()
    for d_data in global_data.values():
        model_keys_in_data.update(d_data.keys())
    model_keys_ordered = [m for m in MODEL_ORDER if any(k.lower() == m.lower() for k in model_keys_in_data)]
    if not model_keys_ordered:
        model_keys_ordered = sorted(model_keys_in_data, key=lambda x: x.lower())
    n_models = len(model_keys_ordered)
    n_sets = len(datasets)
    if n_sets == 0 or n_models == 0:
        return
    width = 0.8 / n_models
    x = np.arange(n_sets)
    fig, ax = plt.subplots(figsize=(max(8, n_sets * 0.8), 5))
    for i, model_key in enumerate(model_keys_ordered):
        scores = []
        for ds in datasets:
            d_data = global_data[ds]
            # match model (case-insensitive)
            m_key = next((k for k in d_data if k.lower() == model_key.lower()), None)
            if m_key is None:
                scores.append(np.nan)
                continue
            vals = []
            for shot_list in d_data[m_key].values():
                v = _mean_ap(shot_list)
                if not np.isnan(v):
                    vals.append(v)
            scores.append(np.mean(vals) if vals else np.nan)
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, scores, width, label=model_key.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel("BBox AP @ IoU=0.50:0.95")
    ax.set_title("Accuracy per model per dataset (aggregated across shots)")
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_root / "summary_model_per_dataset.png", dpi=300)
    plt.close(fig)


# Colors and styles for model-vs-shots (camera-ready): our methods vs baselines
_MODEL_VS_SHOTS_COLORS = {
    "dinov2_l": "#0173b2",      # blue
    "dinov3_l": "#029e73",      # teal/green
    "dinov3_sat_l": "#de8f05",  # amber
    "DETIC": "#cc78bc",         # mauve
    "DEViT": "#ca9161",         # brown
}
# Baselines use dashed lines so they're visually distinct from training-free
_MODEL_VS_SHOTS_LINESTYLE = {
    "DETIC": "--",
    "DEViT": "--",
}
# Legend labels for paper (training-free names match table)
_MODEL_VS_SHOTS_LEGEND_LABELS = {
    "dinov2_l": "Training-free (DINOv2-L)",
    "dinov3_l": "Training-free (DINOv3-L)",
    "dinov3_sat_l": "Training-free (DINOv3-Sat-L)",
    "DETIC": "DETIC",
    "DEViT": "DEViT",
}


def plot_summary_model_vs_shots(global_data, output_root):
    """Accuracy per model vs shots (line), aggregated across datasets. Camera-ready:
    no error bars or std bands; legend ordered by performance at max shots."""
    all_shot_vals = _all_shots(global_data)
    if not all_shot_vals:
        return
    model_keys_in_data = set()
    for d_data in global_data.values():
        model_keys_in_data.update(d_data.keys())
    model_keys_ordered = [m for m in MODEL_ORDER if any(k.lower() == m.lower() for k in model_keys_in_data)]
    if not model_keys_ordered:
        model_keys_ordered = sorted(model_keys_in_data, key=lambda x: x.lower())

    # Build means and stds per model (stds only for optional fill)
    series = {}
    for model_key in model_keys_ordered:
        means, stds = [], []
        for shot in all_shot_vals:
            vals = []
            for d_data in global_data.values():
                m_key = next((k for k in d_data if k.lower() == model_key.lower()), None)
                if m_key is None or shot not in d_data[m_key]:
                    continue
                v = _mean_ap(d_data[m_key][shot])
                if not np.isnan(v):
                    vals.append(v)
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        series[model_key] = {"means": np.array(means), "stds": np.array(stds)}

    # Legend order: by mean at highest shot (best first) so "who's better" is clear
    last_shot = all_shot_vals[-1]
    last_shot_idx = all_shot_vals.index(last_shot)
    model_order_by_perf = sorted(
        model_keys_ordered,
        key=lambda m: series[m]["means"][last_shot_idx] if not np.isnan(series[m]["means"][last_shot_idx]) else -1,
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.array(all_shot_vals)

    for model_key in model_order_by_perf:
        means = series[model_key]["means"]
        color = _MODEL_VS_SHOTS_COLORS.get(model_key, None)
        ls = _MODEL_VS_SHOTS_LINESTYLE.get(model_key, "-")
        label = _MODEL_VS_SHOTS_LEGEND_LABELS.get(model_key, model_key.replace("_", " "))

        # # Light variance band (no whiskers): shows spread without clutter
        # valid = ~np.isnan(means)
        # if valid.any() and np.any(stds[valid] > 0):
        #     ax.fill_between(
        #         x[valid],
        #         np.maximum(0, means[valid] - stds[valid]),
        #         np.minimum(1, means[valid] + stds[valid]),
        #         color=color or "gray",
        #         alpha=0.2,
        #         linewidth=0,
        #     )
        ax.plot(
            x,
            means,
            color=color,
            linestyle=ls,
            linewidth=2.8,
            marker="o",
            markersize=8,
            markeredgewidth=1.2,
            markeredgecolor="white",
            label=label,
            zorder=10,
        )

    ax.set_xlabel("Number of shots")
    ax.set_ylabel("BBox AP @ IoU=0.50:0.95")
    ax.set_title("Accuracy per model vs shots (aggregated across datasets)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85), framealpha=0.95, fontsize=11)
    ax.grid(True, axis="y", alpha=0.45, linestyle="-")
    ax.grid(True, axis="x", alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    if all_shot_vals:
        ax.set_xticks(all_shot_vals)
    fig.tight_layout()
    fig.savefig(output_root / "summary_model_vs_shots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) / "accuracy"
    output_root.mkdir(parents=True, exist_ok=True)

    skipped = []
    # dataset -> model -> shot -> list of metric dicts
    global_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dataset_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name

        # model -> shot -> list of values
        values = defaultdict(lambda: defaultdict(list))

        for exp_dir in dataset_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            m = SHOT_RE.fullmatch(exp_dir.name)
            if not m:
                print(f"Skipping {exp_dir} because it doesn't match the SHOT_RE pattern")
                continue

            shot = int(m.group("shot"))
            model = m.group("model")

            stats_path = exp_dir / "coco_eval_stats_.txt"
            if not stats_path.exists():
                print(f"Skipping {exp_dir} because it doesn't have a coco_eval_stats_.txt file")
                skipped.append(exp_dir)
                continue

            metrics = parse_coco_eval(stats_path)
            if metrics is None:
                skipped.append(exp_dir)
                continue

            values[model][shot].append(metrics)
            global_data[dataset][model][shot].append(metrics)

        if not values:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

        plot_specs = [
            ("bbox_50_95", r"BBox AP @ IoU=0.50:0.95"),
            ("bbox_50", r"BBox AP @ IoU=0.50"),
        ]

        for ax, (key, title) in zip(axes, plot_specs):
            for model in MODEL_ORDER:
                # Check if model is present either as supplied or as lower/upper case variant in values keys
                # (E.g. if someone used detic instead of DETIC)
                available_models = {k.lower(): k for k in values.keys()}
                model_key = available_models.get(model.lower())
                if not model_key:
                    continue

                shots = sorted(values[model_key].keys())
                means, stds = [], []

                for s in shots:
                    # Only plot the key if it exists in all values for this shot
                    vals = [v.get(key) for v in values[model_key][s]]
                    vals = [v for v in vals if v is not None]
                    if not vals:
                        means.append(np.nan)
                        stds.append(0)
                    else:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals))

                ax.errorbar(
                    shots,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=3,
                    label=model.replace("_", " "),
                )

            ax.set_title(title)
            ax.set_xlabel("Number of shots")
            ax.set_ylabel("AP")
            ax.grid(True)

        axes[1].legend()
        fig.suptitle(dataset)
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        fig.savefig(output_root / f"{dataset}.png", dpi=300)
        plt.close(fig)

    # -----------------------------
    # Summary plots (global)
    # -----------------------------
    if global_data:
        plot_summary_aggregated_vs_shots(global_data, output_root)
        plot_summary_overall_by_model(global_data, output_root)
        plot_summary_overall_by_dataset(global_data, output_root)
        plot_summary_model_per_dataset(global_data, output_root)
        plot_summary_model_vs_shots(global_data, output_root)

    # -----------------------------
    # Report skipped experiments
    # -----------------------------
    if skipped:
        print(f"\n{ORANGE}Skipped experiments (missing/incomplete files):{RESET}")
        for p in skipped:
            print(f"{ORANGE}  - {p}{RESET}")


if __name__ == "__main__":
    main()
