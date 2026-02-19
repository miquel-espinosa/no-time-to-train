#!/usr/bin/env python3
"""
Generate LaTeX table for EO bbox results (mAP). Reads coco_eval_stats_.txt from
experiment folders and fills the table, bolding best per column per shot and
best average. Missing values are printed as "r" with a terminal warning.

Usage:
  python scripts/paper_figures/table_EO_results.py ./EO_results
"""
import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------------------------
# Global settings (edit to change shot settings or table structure)
# -----------------------------------------------------------------------------
SHOT_SETTINGS = [1, 5, 10]

# Dataset folder name (as found under results-dir) -> table column order.
# Order here must match the LaTeX table header: FA ST, HR SID, ISA ID, ...
DATASET_COLUMN_ORDER = [
    "FAST",           # FA ST
    "HRSID",          # HR SID
    "ISAID",          # ISA ID
    "MAPPING",        # MAP PING
    "NWPU",           # NW PU
    "RAREPLANES",     # RARE PL.
    "RAREPLANES_SINGLE_CLASS",  # RARE S.C.
    "SIOR",           # SI OR
    "SODAA",          # SO DAA
    "SOTA",           # SO TA
    "SSDD",           # SS DD
    "VEDAI1024",      # VDAI 1024
    "VEDAI512",       # VDAI 512
    "XVIEW",          # XV IEW
]
NUM_DATASET_COLUMNS = len(DATASET_COLUMN_ORDER)

# Table row order: (model key as in folder names, LaTeX label for the row)
# Model keys must match SHOT_RE group "model". Order here is the order in the table.
TABLE_ROW_ORDER = [
    ("DEViT", r"DEViT~\citep{devit}"),
    ("DETIC", r"Detic~\citep{detic}"),
    ("dinov3_sat_l", r"\textbf{Training-free} (DINOv3-Sat-L)"),
    ("dinov2_l", r"\textbf{Training-free} (DINOv2-L)"),
    ("dinov3_l", r"\textbf{Training-free} (DINOv3-Web-L)"),
]

# Regex to parse experiment directory names (same as plot_EO_accuracy.py)
SHOT_RE = re.compile(
    r"(?P<shot>\d+)_shot_(?P<model>dinov[23](?:_sat)?_l|DETIC|DEViT)_seed(?P<seed>\d+)"
)
ACCURACY_KEY = "bbox_50_95"
MISSING_PLACEHOLDER = "r"  # LaTeX placeholder for running/missing experiments
WARN = "\033[33m"   # yellow
RESET = "\033[0m"


def parse_coco_eval(stats_path: Path):
    """
    Parse coco_eval_stats_.txt and return dict with bbox_50_95, bbox_50, etc.
    Returns None if file missing or parsing fails.
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


def load_all_results(input_root: Path):
    """
    Scan input_root for dataset dirs and experiment dirs. Return:
      (data, warnings)
    where data[shot][model][dataset] = float (bbox_50_95 mean over seeds) or None if missing,
    and warnings is a list of strings to print.
    """
    input_root = Path(input_root)
    # shot -> model -> dataset -> list of values (over seeds)
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    warnings = []

    # Normalize dataset name to match DATASET_COLUMN_ORDER (uppercase, minus -> underscore)
    def norm_ds(name):
        u = name.upper().replace("-", "_")
        return u

    for dataset_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        dataset_folder_name = dataset_dir.name
        dataset_key = norm_ds(dataset_folder_name)
        if dataset_key not in DATASET_COLUMN_ORDER:
            continue

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
                warnings.append(f"Missing file: {stats_path}")
                continue
            metrics = parse_coco_eval(stats_path)
            if metrics is None:
                warnings.append(f"Could not parse or incomplete: {stats_path}")
                continue
            raw[shot][model][dataset_key].append(metrics[ACCURACY_KEY])

    # Convert to data[shot][model][dataset] = mean or None
    data = defaultdict(lambda: defaultdict(dict))
    for shot, by_model in raw.items():
        for model, by_ds in by_model.items():
            for ds, vals in by_ds.items():
                if vals:
                    data[shot][model][ds] = sum(vals) / len(vals)
                else:
                    data[shot][model][ds] = None

    return data, warnings


def build_table_cells(data, shot):
    """
    For the given shot, build a matrix: list of rows, each row is list of cell values.
    Cell is either float, or None (meaning missing -> "r").
    Also return per-column best indices (which row index is best for each col).
    """
    rows = []
    for model_key, _ in TABLE_ROW_ORDER:
        row = []
        for ds in DATASET_COLUMN_ORDER:
            val = data.get(shot, {}).get(model_key, {}).get(ds)
            row.append(val)
        # Average over available dataset values
        numeric = [v for v in row if v is not None]
        avg = sum(numeric) / len(numeric) if numeric else None
        row.append(avg)
        rows.append(row)
    return rows


def best_per_column(rows):
    """Return list of row indices (one per column) that have the max value. Last column is Avg."""
    ncols = len(rows[0]) if rows else 0
    best = []
    for c in range(ncols):
        col_vals = [(i, rows[i][c]) for i in range(len(rows)) if rows[i][c] is not None]
        if not col_vals:
            best.append(None)
            continue
        best.append(max(col_vals, key=lambda x: x[1])[0])
    return best


def format_cell(val, bold):
    if val is None:
        return MISSING_PLACEHOLDER
    # Display in 0-100 format (metric is stored as 0-1)
    s = f"{val * 100:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def emit_latex_table(data, warnings):
    """Print full LaTeX table to stdout and warnings to stderr."""
    for w in warnings:
        print(f"{WARN}[WARNING] {w}{RESET}", file=sys.stderr)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\addtolength{\tabcolsep}{-0.4em} % More relaxed spacing now that we have fewer columns")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(r"\begin{adjustbox}{width=\linewidth}")
    lines.append(r"\begin{tabular}{l" + "c" * (NUM_DATASET_COLUMNS + 1) + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\multicolumn{1}{c}{\textbf{Method}} & "
        + " & ".join(
            [
                r"\textbf{\makecell{FA \\ ST}}",
                r"\textbf{\makecell{HR \\ SID}}",
                r"\textbf{\makecell{ISA \\ ID}}",
                r"\textbf{\makecell{MAP \\ PING}}",
                r"\textbf{\makecell{NW \\ PU}}",
                r"\textbf{\makecell{RARE \\ PL.}}",
                r"\textbf{\makecell{RARE \\ S.C.}}",
                r"\textbf{\makecell{SI \\ OR}}",
                r"\textbf{\makecell{SO \\ DAA}}",
                r"\textbf{\makecell{SO \\ TA}}",
                r"\textbf{\makecell{SS \\ DD}}",
                r"\textbf{\makecell{VDAI \\ 1024}}",
                r"\textbf{\makecell{VDAI \\ 512}}",
                r"\textbf{\makecell{XV \\ IEW}}",
                r"\textbf{Avg}",
            ]
        )
        + r" \\"
    )
    lines.append(r"\toprule")
    lines.append(r"\toprule")

    for shot in SHOT_SETTINGS:
        lines.append(r"\multicolumn{16}{c}{\textbf{" + f"{shot}-shot" + r"}} \\")
        lines.append(r"\midrule")
        rows = build_table_cells(data, shot)
        best_col = best_per_column(rows)
        for r, (model_key, latex_label) in enumerate(TABLE_ROW_ORDER):
            cells = []
            for c in range(len(rows[r])):
                val = rows[r][c]
                if val is None:
                    col_name = DATASET_COLUMN_ORDER[c] if c < NUM_DATASET_COLUMNS else "Avg"
                    print(
                        f"{WARN}[WARNING] Missing value: {shot}-shot, {model_key}, {col_name} -> writing 'r'{RESET}",
                        file=sys.stderr,
                    )
                bold = best_col[c] == r
                cells.append(format_cell(val, bold))
            line = latex_label + " & " + " & ".join(cells) + r" \\"
            lines.append(line)
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(
        r"\caption{Cross-domain few-shot object detection performance (mAP) for bounding box results across 14 specialized datasets. We compare other training-free methods against our approach.}"
    )
    lines.append(r"\label{tab:bbox_cdfsod}")
    lines.append(r"\end{table}")

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table for EO bbox results from experiment folder."
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Folder containing per-dataset subdirs (e.g. FAST, HRSID, ...) with experiment dirs like 1_shot_dinov2_l_seed0",
    )
    args = parser.parse_args()
    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"Not a directory: {results_path}", file=sys.stderr)
        sys.exit(1)
    data, warnings = load_all_results(results_path)
    emit_latex_table(data, warnings)


if __name__ == "__main__":
    main()
