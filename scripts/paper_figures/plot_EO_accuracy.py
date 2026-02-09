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
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
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
]
SHOT_RE = re.compile(
    r"(?P<shot>\d+)_shot_(?P<model>dinov[23](?:_sat)?_l)_seed(?P<seed>\d+)"
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
    """
    metrics = {}
    section = None

    try:
        with stats_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s.startswith("===== BBOX RESULTS"):
                    section = "bbox"
                elif s.startswith("===== SEGM RESULTS"):
                    section = "segm"
                elif section and s.startswith("AP IoU=0.50:0.95:"):
                    metrics[f"{section}_50_95"] = float(s.split(":")[-1])
                elif section and s.startswith("AP IoU=0.50:"):
                    metrics[f"{section}_50"] = float(s.split(":")[-1])
    except Exception:
        return None

    required = {"bbox_50_95", "bbox_50", "segm_50_95", "segm_50"}
    if not required.issubset(metrics):
        return None

    return metrics


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

    for dataset_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name

        # model -> shot -> list of values
        values = defaultdict(lambda: defaultdict(list))

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
                skipped.append(exp_dir)
                continue

            metrics = parse_coco_eval(stats_path)
            if metrics is None:
                skipped.append(exp_dir)
                continue

            values[model][shot].append(metrics)

        if not values:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)

        plot_specs = [
            ("bbox_50_95", r"BBox AP @ IoU=0.50:0.95"),
            ("bbox_50", r"BBox AP @ IoU=0.50"),
        ]

        for ax, (key, title) in zip(axes, plot_specs):
            for model in MODEL_ORDER:
                if model not in values:
                    continue

                shots = sorted(values[model].keys())
                means, stds = [], []

                for s in shots:
                    vals = [v[key] for v in values[model][s]]
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
    # Report skipped experiments
    # -----------------------------
    if skipped:
        print(f"\n{ORANGE}Skipped experiments (missing/incomplete files):{RESET}")
        for p in skipped:
            print(f"{ORANGE}  - {p}{RESET}")


if __name__ == "__main__":
    main()
