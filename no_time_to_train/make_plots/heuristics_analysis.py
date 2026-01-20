#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from pycocotools.coco import COCO

# ---------------------------------------------------------------------
# Repo root detection
# ---------------------------------------------------------------------
def find_repo_root(start: Optional[Path] = None) -> Path:
    start = start or Path.cwd()
    current = start.resolve()
    for _ in range(8):
        if (current / "data").exists() and (current / "work_dirs").exists():
            return current
        if (current.parent == current):
            break
        current = current.parent
    return (Path.cwd() / "../../").resolve()

REPO_ROOT = find_repo_root()
PER_CLASS_DIR = REPO_ROOT / "data/coco/annotations/per_class_instances"
ABLATION_WORK_DIR = REPO_ROOT / "work_dirs/1shot_ref_ablation"

# ---------------------------------------------------------------------
# Meta-info (few-shot classes)
# ---------------------------------------------------------------------
from no_time_to_train.dataset.metainfo import METAINFO
COCO_CLASSES: List[str] = list(METAINFO["few_shot_classes"])

# Put 'bird' last
COCO_CLASSES.remove("bird")
COCO_CLASSES.append("bird")

# ---------------------------------------------------------------------
# Helper to convert class names to folder-safe format
# ---------------------------------------------------------------------
def class_to_folder_name(class_name: str) -> str:
    """Convert class name to folder-safe format (spaces to underscores)."""
    return class_name.replace(" ", "_")

# ---------------------------------------------------------------------
# CSV parsing for reference image selections
# ---------------------------------------------------------------------
def parse_ref_selection_csv(class_name: str) -> pd.DataFrame:
    folder_name = class_to_folder_name(class_name)
    csv_path = PER_CLASS_DIR / folder_name / "instances" / "ref_image_selections.csv"
    if not csv_path.exists():
        return pd.DataFrame(
            columns=["ann_id", "area_cat", "centered", "avoid_sides"]
        ).astype({"ann_id": "Int64", "area_cat": "string", "centered": "boolean", "avoid_sides": "Int64"})

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"id": "ann_id", "area": "area_cat"})

    df["ann_id"] = pd.to_numeric(df.get("ann_id"), errors="coerce").astype("Int64")
    df["area_cat"] = df.get("area_cat", pd.Series(dtype="string")).astype("string").str.lower()
    df["centered"] = (
        df.get("centered", pd.Series(dtype="boolean"))
        .astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .astype("boolean")
    )
    df["avoid_sides"] = pd.to_numeric(df.get("avoid_sides"), errors="coerce").astype("Int64")
    df["class"] = class_name

    keep = ["class", "ann_id", "area_cat", "centered", "avoid_sides"]
    return df[keep]

# ---------------------------------------------------------------------
# Parsing score files (bbox/segm AP)
# ---------------------------------------------------------------------
def parse_scores_for_ref(class_name: str, ann_id: int) -> Tuple[Optional[float], Optional[float]]:
    folder_name = class_to_folder_name(class_name)
    exp_dir = ABLATION_WORK_DIR / f"{folder_name}_{ann_id}"
    stats_path = exp_dir / "coco_eval_stats_.txt"
    if not stats_path.exists():
        print(f"Missing stats for {class_name} {ann_id}")
        return None, None

    bbox_ap = None
    segm_ap = None
    section = None

    try:
        with stats_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("===== BBOX RESULTS ====="):
                    section = "bbox"
                    continue
                if s.startswith("===== SEGM RESULTS ====="):
                    section = "segm"
                    continue
                if section in {"bbox", "segm"} and s.startswith("AP IoU=0.50:0.95:"):
                    value_str = s.split(":", maxsplit=2)[-1].strip()
                    try:
                        val = float(value_str)
                    except Exception:
                        raise ValueError(f"Failed to parse {value_str} as float")

                    if section == "bbox":
                        bbox_ap = val
                    else:
                        segm_ap = val
    except Exception:
        raise Exception("Failed to parse scores for ref")

    return bbox_ap, segm_ap

# ---------------------------------------------------------------------
# COCO loading and feature extraction
# ---------------------------------------------------------------------
def load_coco_for_class(class_name: str) -> Optional[COCO]:
    folder_name = class_to_folder_name(class_name)
    ann_path = PER_CLASS_DIR / folder_name / "instances_train2017.json"
    if not ann_path.exists():
        return None
    try:
        return COCO(str(ann_path))
    except Exception:
        raise Exception("Failed to load COCO for class")

def compute_ann_features(coco: COCO, ann_id: int) -> Optional[Dict[str, float]]:
    anns = coco.loadAnns([int(ann_id)])
    if not anns:
        return None
    ann = anns[0]

    img = coco.loadImgs([int(ann.get("image_id"))])[0]
    iw, ih = float(img["width"]), float(img["height"])

    area = float(ann.get("area", 0.0))
    norm_area = area / (iw * ih)

    x, y, w, h = map(float, ann.get("bbox", [0,0,0,0]))
    cx = x + w/2
    cy = y + h/2

    left = x
    top = y
    right = iw - (x + w)
    bottom = ih - (y + h)
    edge = min(left, top, right, bottom)

    return {
        "abs_area_px": area,
        "norm_area_px": norm_area,
        "norm_bbox_x": cx / iw,
        "norm_bbox_y": cy / ih,
        "min_dist_to_edge_px": edge,
    }

# ---------------------------------------------------------------------
# Build dataset and plotting – script entry
# ---------------------------------------------------------------------
def build_reference_dataset() -> pd.DataFrame:
    """
    Build a per-reference dataset combining selection CSVs, COCO-derived features,
    and evaluation scores. Saves the CSV to ABLATION_WORK_DIR/analysis and
    returns the DataFrame.
    """
    all_rows: List[Dict] = []

    classes_to_process = [c for c in COCO_CLASSES if (PER_CLASS_DIR / class_to_folder_name(c)).exists()]
    print(f"Found {len(classes_to_process)} COCO few-shot classes with per-class data.")

    for class_name in tqdm(classes_to_process, desc="Classes"):
        sel_df = parse_ref_selection_csv(class_name)
        if sel_df.empty:
            continue

        coco = load_coco_for_class(class_name)
        for _, row in sel_df.iterrows():
            ann_id = int(row["ann_id"]) if pd.notna(row["ann_id"]) else None
            if ann_id is None:
                continue

            bbox_ap, segm_ap = parse_scores_for_ref(class_name, ann_id)

            feats = {
                "class": class_name,
                "ann_id": ann_id,
                "area_cat": row["area_cat"] if pd.notna(row["area_cat"]) else None,
                "centered": (bool(row["centered"]) if pd.notna(row["centered"]) else None),
                "avoid_sides": (int(row["avoid_sides"]) if pd.notna(row["avoid_sides"]) else None),
                "bbox_ap_iou": bbox_ap,
                "segm_ap_iou": segm_ap,
            }

            ann_feats = compute_ann_features(coco, ann_id) if coco else None
            if ann_feats:
                feats.update(ann_feats)
            else:
                feats.update({
                    "abs_area_px": None,
                    "norm_area_px": None,
                    "norm_bbox_x": None,
                    "norm_bbox_y": None,
                    "min_dist_to_edge_px": None,
                })

            all_rows.append(feats)

    columns = [
        "class","ann_id","area_cat","centered","avoid_sides",
        "bbox_ap_iou","segm_ap_iou","abs_area_px","norm_area_px",
        "norm_bbox_x","norm_bbox_y","min_dist_to_edge_px",
    ]
    dataset_df = pd.DataFrame(all_rows, columns=columns)

    analysis_dir = ABLATION_WORK_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "1shot_ref_dataset.csv"
    dataset_df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path}")
    return dataset_df


def prepare_processed_df(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a filtered and enriched DataFrame with:
      - class-normalized scores
      - bbox relative width/height within image
    Prints the number of filtered rows for traceability.
    """
    processed_df = dataset_df.dropna(subset=["bbox_ap_iou", "segm_ap_iou"]).copy()
    print(f"Filtered rows: {len(processed_df)} / {len(dataset_df)}")

    def _minmax_series(s: pd.Series) -> pd.Series:
        minv, maxv = s.min(), s.max()
        if pd.isna(minv) or pd.isna(maxv):
            return pd.Series([np.nan]*len(s), index=s.index)
        rng = maxv - minv
        if rng <= 1e-12:
            return pd.Series([0.5]*len(s), index=s.index)
        return (s - minv) / rng

    processed_df["norm_bbox_ap_iou"] = processed_df.groupby("class")["bbox_ap_iou"].transform(_minmax_series)
    processed_df["norm_segm_ap_iou"] = processed_df.groupby("class")["segm_ap_iou"].transform(_minmax_series)

    # Compute relative width/height per annotation
    _coco_cache: Dict[str, Optional[COCO]] = {}

    def _get_coco(class_name: str) -> Optional[COCO]:
        if class_name not in _coco_cache:
            _coco_cache[class_name] = load_coco_for_class(class_name)
        return _coco_cache[class_name]

    def _compute_wh_norm(row: pd.Series) -> Tuple[float, float]:
        cls = row["class"]
        ann_id = int(row["ann_id"])
        coco = _get_coco(cls)
        if coco is None:
            return (np.nan, np.nan)
        anns = coco.loadAnns([ann_id])
        if not anns:
            return (np.nan, np.nan)
        ann = anns[0]
        img = coco.loadImgs([ann["image_id"]])[0]
        iw, ih = float(img["width"]), float(img["height"])
        x, y, w, h = map(float, ann.get("bbox", [np.nan]*4))
        return (w/iw, h/ih)

    wh = [_compute_wh_norm(r) for _, r in processed_df.iterrows()]
    processed_df["norm_bbox_w"] = [t[0] for t in wh]
    processed_df["norm_bbox_h"] = [t[1] for t in wh]

    return processed_df


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, filename: str, plots_dir: Path, bins: int = 40) -> None:
    """
    Plot a 2D heatmap of a value aggregated over binned bbox relative width/height.
    """
    d = df[["norm_bbox_w", "norm_bbox_h", value_col]].dropna()
    d = d[(d["norm_bbox_w"] >= 0)&(d["norm_bbox_w"]<=1)&(d["norm_bbox_h"]>=0)&(d["norm_bbox_h"]<=1)]
    if d.empty:
        print(f"No data for heatmap: {value_col}")
        return
    d["w_bin"] = pd.cut(d["norm_bbox_w"], bins=bins, labels=False, include_lowest=True)
    d["h_bin"] = pd.cut(d["norm_bbox_h"], bins=bins, labels=False, include_lowest=True)
    pivot = d.groupby(["h_bin","w_bin"], observed=True)[value_col].mean().unstack("w_bin")

    plt.figure(figsize=(8,6))
    sns.heatmap(pivot.iloc[::-1], vmin=0, vmax=1, cmap="viridis")
    plt.title(title)
    plt.xlabel("bbox relative width (bin)")
    plt.ylabel("bbox relative height (bin)")
    plt.tight_layout()
    plt.savefig(plots_dir / filename, dpi=200)
    plt.close()


def generate_plots(dataset_df: pd.DataFrame) -> None:
    """
    Generate analysis plots to the fixed plots directory under the repo.
    """
    sns.set_theme(style="whitegrid", context="paper")

    processed_df = prepare_processed_df(dataset_df)

    plots_dir = REPO_ROOT / "no_time_to_train/make_plots/heuristics_analysis"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Heatmaps
    plot_heatmap(
        processed_df,
        "norm_bbox_ap_iou",
        "Avg class-normalized BBOX score by bbox rel. width/height",
        "heatmap_bbox_norm_scores.png",
        plots_dir,
    )

    plot_heatmap(
        processed_df,
        "norm_segm_ap_iou",
        "Avg class-normalized SEGM score by bbox rel. width/height",
        "heatmap_segm_norm_scores.png",
        plots_dir,
    )

    # Center-position heatmaps (bbox center x/y)
    def plot_center_heatmap(df: pd.DataFrame, value_col: str, title: str, filename: str, bins: int = 40, kde_smooth: bool = False) -> None:
        d = df[["norm_bbox_x", "norm_bbox_y", value_col]].dropna()
        d = d[(d["norm_bbox_x"] >= 0)&(d["norm_bbox_x"]<=1)&(d["norm_bbox_y"]>=0)&(d["norm_bbox_y"]<=1)]
        if d.empty:
            print(f"No data for center heatmap: {value_col}")
            return
        
        # Bin the data to compute average scores per bin
        d["x_bin"] = pd.cut(d["norm_bbox_x"], bins=bins, labels=False, include_lowest=True)
        d["y_bin"] = pd.cut(d["norm_bbox_y"], bins=bins, labels=False, include_lowest=True)
        pivot = d.groupby(["y_bin","x_bin"], observed=True)[value_col].mean().unstack("x_bin")
        
        if kde_smooth:
            # Apply weighted smoothing for continuous-looking heatmap
            from scipy.ndimage import gaussian_filter
            
            # Compute sum and count pivots for weighted averaging
            pivot_sum = d.groupby(["y_bin","x_bin"], observed=True)[value_col].sum().unstack("x_bin")
            pivot_count = d.groupby(["y_bin","x_bin"], observed=True)[value_col].count().unstack("x_bin")
            
            # Fill NaNs with 0 (representing no data contribution)
            # This is mathematically correct: bins with no data contribute zero to both sum and count
            pivot_sum_filled = pivot_sum.fillna(0.0)
            pivot_count_filled = pivot_count.fillna(0.0)
            
            # Smooth both sum and count separately with Gaussian filter
            # Tunable parameter: sigma controls smoothness (1.0-3.0 typical range)
            sigma = 1.5
            smoothed_sum = gaussian_filter(pivot_sum_filled.values, sigma=sigma)
            smoothed_count = gaussian_filter(pivot_count_filled.values, sigma=sigma)
            
            # Calculate smoothed mean by dividing smoothed sum by smoothed count
            # This produces a locally-weighted average that is unbiased
            with np.errstate(divide='ignore', invalid='ignore'):
                smoothed_mean = smoothed_sum / smoothed_count
                # Mask regions with negligible count (boundary regions with insufficient data)
                # Tunable parameter: threshold relative to max count (0.05-0.2 typical range)
                smoothed_mean[smoothed_count < 0.1] = np.nan
            
            # Create the plot with smoothing
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot smoothed heatmap without grid lines
            # Using sequential colormap (viridis) for perceptual accuracy with magnitude data
            im = ax.imshow(
                smoothed_mean[::-1],  # Flip vertically to match coordinate system
                extent=[0, bins, 0, bins],
                aspect='auto',
                cmap="RdYlGn",  # Sequential colormap for 0-1 magnitude data
                vmin=0,
                vmax=1,
                interpolation='bilinear'
            )
            
            # Add contour lines to delineate regions (only where we have data)
            contour_levels = np.linspace(0.2, 0.9, 8)
            X, Y = np.meshgrid(np.arange(bins), np.arange(bins))
            # Use the masked smoothed data for contours
            contours = ax.contour(
                X, Y,
                smoothed_mean[::-1],
                levels=contour_levels,
                colors='black',
                alpha=0.3,
                linewidths=0.8
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Score', rotation=90, labelpad=15)
            
            # Remove tick labels but keep the axes clean
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.set_title(title, pad=10)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            # Ensure frame is visible for both heatmap and colorbar
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_edgecolor('gray')
            
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            # Simple discrete heatmap (original version)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(pivot.iloc[::-1], vmin=0, vmax=1, cmap="RdYlGn", xticklabels=False, yticklabels=False, ax=ax, cbar_kws={'label': 'Score'})
            ax.set_title(title)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            
            # Ensure frame is visible for both heatmap and colorbar
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_edgecolor('gray')
            
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=200)
            plt.close()

    plot_center_heatmap(
        processed_df,
        "norm_bbox_ap_iou",
        "BBOX score by center position",
        "heatmap_center_bbox_norm_scores_kde_smooth.png",
        bins=30,
        kde_smooth=True,
    )
    plot_center_heatmap(
        processed_df,
        "norm_bbox_ap_iou",
        "BBOX score by center position",
        "heatmap_center_bbox_norm_scores.png",
        bins=30,
        kde_smooth=False,
    )
    plot_center_heatmap(
        processed_df,
        "norm_segm_ap_iou",
        "SEGM score by center position",
        "heatmap_center_segm_norm_scores_kde_smooth.png",
        bins=30,
        kde_smooth=True,
    )
    plot_center_heatmap(
        processed_df,
        "norm_segm_ap_iou",
        "SEGM score by center position",
        "heatmap_center_segm_norm_scores.png",
        bins=30,
        kde_smooth=False,
    )

    # Per-class scatter
    # Add sanitized class labels (replace spaces) for robust faceting
    processed_df = processed_df.copy()
    processed_df["class_safe"] = processed_df["class"].astype(str).str.replace(" ", "_")
    raw_long = processed_df.melt(
        id_vars=["class_safe","norm_area_px","ann_id"],
        value_vars=["bbox_ap_iou","segm_ap_iou"],
        var_name="metric", value_name="score"
    )
    if not raw_long.empty:
        # consistent palette for scatter and regression lines
        _metric_order_raw = ["bbox_ap_iou","segm_ap_iou"]
        _palette_raw_list = sns.color_palette("tab10", len(_metric_order_raw))
        _metric_palette_raw = dict(zip(_metric_order_raw, _palette_raw_list))

        # Find best/worst examples for airplane and bird
        special_classes = ["airplane", "bird"]
        special_points = {}
        
        for cls in special_classes:
            cls_data = processed_df[processed_df["class"] == cls].copy()
            if len(cls_data) > 0:
                # Use bbox_ap_iou as the metric for selection
                cls_data_sorted = cls_data.sort_values("bbox_ap_iou")
                # Pick the 20th worst (or closest available if fewer samples)
                worst_idx = min(19, len(cls_data_sorted) - 1)
                worst = cls_data_sorted.iloc[worst_idx]
                best = cls_data_sorted.iloc[-1]
                
                special_points[cls] = {
                    "worst": {"ann_id": worst["ann_id"], "score": worst["bbox_ap_iou"], "area": worst["norm_area_px"]},
                    "best": {"ann_id": best["ann_id"], "score": best["bbox_ap_iou"], "area": best["norm_area_px"]}
                }
                
                print(f"\n{cls.upper()} special examples:")
                print(f"  Worst: ann_id={worst['ann_id']}, bbox_ap_iou={worst['bbox_ap_iou']:.4f}, area={worst['norm_area_px']:.6f}")
                print(f"  Best:  ann_id={best['ann_id']}, bbox_ap_iou={best['bbox_ap_iou']:.4f}, area={best['norm_area_px']:.6f}")

        g = sns.FacetGrid(
            raw_long,
            col="class_safe",
            col_wrap=5,
            sharex=True,
            sharey=False,
            hue="metric",
            height=2.2,
            palette=_metric_palette_raw,
        )
        g.map_dataframe(sns.scatterplot, x="norm_area_px", y="score", alpha=0.55, s=14, linewidth=0)
        g.add_legend(title="metric", adjust_subtitles=True, loc="upper left", labels=["bbox", "segm"])
        g._legend.set_frame_on(True)
        g._legend.get_frame().set_facecolor('white')
        g._legend.get_frame().set_alpha(1.0)
        g.set_axis_labels("Normalised Mask Area (log)","Score")
        g.set(xscale="log")
        g.set_titles("{col_name}")
        g.figure.subplots_adjust(wspace=0.15, hspace=0.25)
        # sns.despine(trim=True)
        # add per-metric log-linear regression lines per facet
        def _add_log_reg_lines(data, color=None, **kwargs):
            ax = plt.gca()
            for m in _metric_order_raw:
                d = data[data["metric"] == m].dropna(subset=["norm_area_px","score"])
                d = d[d["norm_area_px"] > 0]
                if len(d) < 2:
                    continue
                x = d["norm_area_px"].to_numpy(dtype=float)
                y = d["score"].to_numpy(dtype=float)
                logx = np.log(x)
                try:
                    b, a = np.polyfit(logx, y, 1)  # y = a + b*log(x)
                except Exception:
                    continue
                x_line = np.geomspace(x.min(), x.max(), 100)
                y_line = a + b * np.log(x_line)
                ax.plot(x_line, y_line, color=_metric_palette_raw[m], linewidth=1.6, alpha=0.9)

        g.map_dataframe(_add_log_reg_lines)
        
        # Add special markers for airplane and bird
        for cls in special_classes:
            if cls not in special_points:
                continue
            cls_safe = cls.replace(" ", "_")
            # Find the corresponding axis
            for ax in g.axes.flat:
                if ax.get_title() == cls_safe:
                    worst = special_points[cls]["worst"]
                    best = special_points[cls]["best"]
                    # Plot red circle for worst
                    ax.scatter([worst["area"]], [worst["score"]], marker='o', s=100, 
                              facecolors='none', edgecolors='red', linewidths=2.5, zorder=10)
                    # Plot green circle for best
                    ax.scatter([best["area"]], [best["score"]], marker='o', s=100, 
                              facecolors='none', edgecolors='green', linewidths=2.5, zorder=10)
                    break
        
        g.figure.tight_layout(pad=0.5)
        g.figure.savefig(plots_dir / "per_class_area_vs_raw_scores.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Aggregated scatter
    norm_long = processed_df.melt(
        id_vars=["norm_area_px"],
        value_vars=["norm_bbox_ap_iou","norm_segm_ap_iou"],
        var_name="metric", value_name="score"
    )
    if not norm_long.empty:
        plt.figure(figsize=(6,5))
        _metric_order_norm = ["norm_bbox_ap_iou","norm_segm_ap_iou"]
        _palette_norm_list = sns.color_palette("tab10", len(_metric_order_norm))
        _metric_palette_norm = dict(zip(_metric_order_norm, _palette_norm_list))
        sns.scatterplot(
            data=norm_long,
            x="norm_area_px",
            y="score",
            hue="metric",
            alpha=0.5,
            s=20,
            palette=_metric_palette_norm,
        )
        plt.ylim(0,1)
        plt.xscale("log")
        plt.xlabel("norm_area_px")
        plt.ylabel("Score (class-normalized)")
        plt.title("All classes: area vs class-normalized scores")
        # add log-linear regression lines per metric
        ax = plt.gca()
        for m in _metric_order_norm:
            d = norm_long[norm_long["metric"] == m].dropna(subset=["norm_area_px","score"])
            d = d[d["norm_area_px"] > 0]
            if len(d) < 2:
                continue
            x = d["norm_area_px"].to_numpy(dtype=float)
            y = d["score"].to_numpy(dtype=float)
            logx = np.log(x)
            try:
                b, a = np.polyfit(logx, y, 1)
            except Exception:
                continue
            x_line = np.geomspace(x.min(), x.max(), 200)
            y_line = a + b * np.log(x_line)
            ax.plot(x_line, y_line, color=_metric_palette_norm[m], linewidth=2, alpha=0.9)
        plt.tight_layout()
        plt.savefig(plots_dir / "all_classes_area_vs_norm_scores.png", dpi=200)
        plt.close()

    # Scatter: min_dist_to_edge_px vs normalized scores
    edge_long = processed_df.melt(
        id_vars=["min_dist_to_edge_px"],
        value_vars=["norm_bbox_ap_iou","norm_segm_ap_iou"],
        var_name="metric", value_name="score"
    )
    if not edge_long.empty:
        plt.figure(figsize=(6,5))
        sns.scatterplot(data=edge_long, x="min_dist_to_edge_px", y="score", hue="metric", alpha=0.5, s=20)
        plt.ylim(0,1)
        # plt.xscale("log")
        plt.xlabel("min_dist_to_edge_px")
        plt.ylabel("Score (class-normalized)")
        plt.title("All classes: min_dist_to_edge_px vs class-normalized scores")
        plt.tight_layout()
        plt.savefig(plots_dir / "edge_distance_vs_norm_scores.png", dpi=200)
        plt.close()

    # Bars: area category
    area_order = ["small","medium","large"]
    agg_area = processed_df.groupby("area_cat")[["norm_bbox_ap_iou","norm_segm_ap_iou"]].mean().reindex(area_order)
    if not agg_area.dropna(how="all").empty:
        x = np.arange(len(agg_area.index))
        width = 0.38
        fig, ax = plt.subplots(figsize=(4,4.5))
        ax.bar(x - width/2, agg_area["norm_bbox_ap_iou"], width, label="bbox")
        ax.bar(x + width/2, agg_area["norm_segm_ap_iou"], width, label="segm")
        ax.set_xticks(x)
        ax.set_xticklabels(agg_area.index)
        ax.set_ylim(0,0.8)
        ax.set_ylabel("Score (class-normalized)")
        ax.set_title("Scores by area category")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "bars_area_category_norm_scores.png", dpi=200)
        plt.close()

    # Bars: centered
    agg_centered = processed_df.groupby("centered")[["norm_bbox_ap_iou","norm_segm_ap_iou"]].mean()
    if not agg_centered.dropna(how="all").empty:
        idx = list(agg_centered.index)
        x = np.arange(len(idx))
        width = 0.38

        fig, ax = plt.subplots(figsize=(3,4.5))
        ax.bar(x - width/2, agg_centered["norm_bbox_ap_iou"], width, label="bbox")
        ax.bar(x + width/2, agg_centered["norm_segm_ap_iou"], width, label="segm")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in idx])
        ax.set_ylim(0,0.8)
        ax.set_ylabel("Score (class-normalized)")
        ax.set_title("Score by position (centered)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "bars_centered_norm_scores.png", dpi=200)
        plt.close()

    # Bars: avoid_sides
    agg_avoid = processed_df.groupby("avoid_sides")[["norm_bbox_ap_iou","norm_segm_ap_iou"]].mean()
    if not agg_avoid.dropna(how="all").empty:
        idx = list(agg_avoid.index)
        x = np.arange(len(idx))
        width = 0.38

        fig, ax = plt.subplots(figsize=(6.5,4.5))
        ax.bar(x - width/2, agg_avoid["norm_bbox_ap_iou"], width, label="bbox")
        ax.bar(x + width/2, agg_avoid["norm_segm_ap_iou"], width, label="segm")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in idx])
        ax.set_ylim(0,1)
        ax.set_ylabel("Score (class-normalized)")
        ax.set_title("Scores by avoid_sides")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "bars_avoid_sides_norm_scores.png", dpi=200)
        plt.close()

    print("Done. Plots saved to:", plots_dir)


def main() -> None:
    dataset_df = build_reference_dataset()
    generate_plots(dataset_df)


if __name__ == "__main__":
    main()
