#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


"""
Usage:
python scripts/paper_figures/plot_EO_runtime.py \
  --input-root ./EO_results \
  --output-root ./EO_results
"""

# -----------------------------
# Styling
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

MODEL_ORDER = [
    "dinov2_l",
    "dinov3_l",
    "dinov3_sat_l",
]

MODEL_SHADE = {
    "dinov2_l": 0.7,
    "dinov3_l": 1.0,
    "dinov3_sat_l": 1.3,
}

SHOT_RE = re.compile(
    r"(?P<shot>\d+)_shot_(?P<model>dinov[23](?:_sat)?_l)_seed(?P<seed>\d+)"
)
GPU_RE = re.compile(r"\((\d+)\s+GPUs?\)")
RUNTIME_RE = re.compile(r"Total runtime\s*:\s*(\d+)s")

ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


def parse_summary(path: Path):
    try:
        txt = path.read_text(errors="ignore")
    except Exception:
        return None
    g = GPU_RE.search(txt)
    r = RUNTIME_RE.search(txt)
    return (int(g.group(1)), int(r.group(1))) if g and r else None

def shade(color, factor):
    r, g, b = mcolors.to_rgb(color)
    return (min(r * factor, 1), min(g * factor, 1), min(b * factor, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) / "runtime"
    output_root.mkdir(parents=True, exist_ok=True)

    skipped = []

    # gpus -> dataset -> model -> shot -> runtimes
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    for dataset_dir in input_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for exp_dir in dataset_dir.iterdir():
            m = SHOT_RE.fullmatch(exp_dir.name)
            if not m:
                continue

            shot = int(m.group("shot"))
            model = m.group("model")

            summary = exp_dir / "summary.txt"
            if not summary.exists():
                skipped.append(exp_dir)
                continue

            parsed = parse_summary(summary)
            if parsed is None:
                skipped.append(exp_dir)
                continue

            gpus, runtime = parsed
            data[gpus][dataset][model][shot].append(runtime)

    cmap = plt.get_cmap("tab10")

    for gpus, gdata in data.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        for d_idx, (dataset, models) in enumerate(sorted(gdata.items())):
            base_color = cmap(d_idx % 10)

            for model in MODEL_ORDER:
                if model not in models:
                    continue

                shots = sorted(models[model].keys())
                means, stds = [], []

                for s in shots:
                    vals = models[model][s]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))

                color = shade(base_color, MODEL_SHADE[model])

                ax.errorbar(
                    shots,
                    means,
                    yerr=stds,
                    marker="o",
                    capsize=3,
                    color=color,
                    linestyle="-" if model == "dinov2_l"
                            else "--" if model == "dinov3_l"
                            else ":",
                    label=f"{dataset} | {model}",
                )


        ax.set_xlabel("Number of shots")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(f"Total runtime ({gpus} GPUs)")
        ax.grid(True)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(output_root / f"{gpus}_runtime.png", dpi=300)
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
