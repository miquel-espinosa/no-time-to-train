import argparse
import json
import os
import re
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def write_json(file_path: str, data: Dict[str, Any]) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_class_name(name: str) -> str:
    name = name.lower().strip()
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def ensure_csv_with_header(csv_path: str, fieldnames: List[str]) -> None:
    if not os.path.exists(csv_path):
        ensure_dir(os.path.dirname(csv_path))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def read_csv_records_by_id(csv_path: str) -> Dict[str, Dict[str, str]]:
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        result: Dict[str, Dict[str, str]] = {}
        for row in reader:
            if not row:
                continue
            rid = str(row.get("id", "")).strip()
            if rid:
                result[rid] = row
        return result


def append_csv_record(csv_path: str, fieldnames: List[str], record: Dict[str, Any]) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(record)


def yellow_warning(msg: str) -> None:
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    print(f"{YELLOW}{msg}{RESET}")


def infer_split_from_filename(path: str) -> str:
    m = re.search(r"(train2017|val2017)", os.path.basename(path))
    return m.group(1) if m else "train2017"


@dataclass
class CocoIndex:
    images_by_id: Dict[int, Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    categories: List[Dict[str, Any]]
    info: Dict[str, Any]
    licenses: List[Dict[str, Any]]


def build_indices(coco: Dict[str, Any]) -> CocoIndex:
    images: List[Dict[str, Any]] = coco.get("images", [])
    annotations: List[Dict[str, Any]] = coco.get("annotations", [])
    categories: List[Dict[str, Any]] = coco.get("categories", [])
    info: Dict[str, Any] = coco.get("info", {})
    licenses: List[Dict[str, Any]] = coco.get("licenses", [])
    images_by_id = {img["id"]: img for img in images}
    return CocoIndex(
        images_by_id=images_by_id,
        annotations=annotations,
        categories=categories,
        info=info,
        licenses=licenses,
    )


def compute_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def centeredness_percent(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float]:
    """
    Returns (cx_pct, cy_pct, dist_norm) where:
      - cx_pct, cy_pct are 0..100 indicating how close to the center along each axis.
      - dist_norm is the radial centeredness in [0, 1], 1 = exactly at center.
    """
    cx, cy = compute_bbox_center(bbox)
    cx_dist = abs(cx - img_w / 2.0)
    cy_dist = abs(cy - img_h / 2.0)
    # 100% when exactly centered, 0% at far edge from center
    cx_pct = max(0.0, 100.0 * (1.0 - cx_dist / (img_w / 2.0))) if img_w > 0 else 0.0
    cy_pct = max(0.0, 100.0 * (1.0 - cy_dist / (img_h / 2.0))) if img_h > 0 else 0.0
    # radial normalized centeredness (0..1)
    # use max of axes extents for normalization robustness
    denom = ((img_w / 2.0) ** 2 + (img_h / 2.0) ** 2) ** 0.5
    dist_norm = max(0.0, 1.0 - ((cx_dist**2 + cy_dist**2) ** 0.5) / denom) if denom > 0 else 0.0
    return cx_pct, cy_pct, dist_norm


def distances_to_edges(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    left = max(0.0, x)
    right = max(0.0, img_w - (x + w))
    top = max(0.0, y)
    bottom = max(0.0, img_h - (y + h))
    return left, right, top, bottom


def plot_analysis(
    coco_path: str,
    output_class_dir: str,
) -> None:
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib and numpy are required for --mode=analysis") from e

    coco = read_json(coco_path)
    idx = build_indices(coco)

    ensure_dir(output_class_dir)

    # 1) Area histogram
    areas = [float(ann.get("area", 0.0)) for ann in idx.annotations]
    if len(areas) > 0:
        plt.figure(figsize=(7, 4))
        plt.hist(areas, bins=50, color="#1f77b4", edgecolor="black")
        plt.xlabel("Annotation area (px^2)")
        plt.ylabel("Count")
        plt.title("Area distribution")
        plt.tight_layout()
        plt.yscale("log")
        plt.savefig(os.path.join(output_class_dir, "area_hist.png"), dpi=150)
        plt.tight_layout()
        plt.close()

    # 2) 2D normalized histogram of centeredness
    # We want actual bbox center positions normalized to [0,1] to align with image layout.
    cx_norms: List[float] = []
    cy_norms: List[float] = []
    for ann in idx.annotations:
        img = idx.images_by_id.get(ann["image_id"])
        if img is None:
            continue
        w = int(img.get("width", 0))
        h = int(img.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        cx, cy = compute_bbox_center(ann["bbox"])
        cx_norms.append(min(max(cx / float(w), 0.0), 1.0))
        cy_norms.append(min(max(cy / float(h), 0.0), 1.0))
    if len(cx_norms) > 0:
        plt.figure(figsize=(6, 5))
        plt.hist2d(cx_norms, cy_norms, bins=[40, 40], range=[[0, 1], [0, 1]], cmap="viridis")
        plt.colorbar(label="Count")
        plt.xlabel("Center X (0 = left, 1 = right)")
        plt.ylabel("Center Y (0 = top, 1 = bottom)")
        # Invert Y to match image coordinates where 0 is top
        ax = plt.gca()
        ax.set_ylim(1, 0)
        ax.set_aspect("equal", adjustable="box")
        plt.title("2D bbox center position histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(output_class_dir, "centeredness_2d_hist.png"), dpi=150)
        plt.tight_layout()
        plt.close()

        # Minimalist square colormap (no axes/labels/colorbar) for grid visualization
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.hist2d(
            cx_norms,
            cy_norms,
            bins=[40, 40],
            range=[[0, 1], [0, 1]],
            cmap="viridis",
        )
        ax2.set_xlim(0, 1)
        ax2.set_ylim(1, 0)  # invert Y to match image coordinates (0 = top)
        ax2.set_aspect("equal", adjustable="box")
        ax2.axis("off")
        plt.savefig(
            os.path.join(output_class_dir, "centeredness_2d_hist_plain.png"),
            dpi=150,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig2)

    # 3) Distance to edges histograms (0..50 px per side)
    DISTANCE_THRESHOLD = 75
    left_d, right_d, top_d, bottom_d = [], [], [], []
    for ann in idx.annotations:
        img = idx.images_by_id.get(ann["image_id"])
        if img is None:
            continue
        w = int(img.get("width", 0))
        h = int(img.get("height", 0))
        l, r, t, b = distances_to_edges(ann["bbox"], w, h)
        left_d.append(l)
        right_d.append(r)
        top_d.append(t)
        bottom_d.append(b)
    if len(left_d) > 0:
        bins = list(range(0, DISTANCE_THRESHOLD + 1))
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes = axes.ravel()
        for ax, data, title in zip(
            axes,
            [left_d, right_d, top_d, bottom_d],
            ["Left", "Right", "Top", "Bottom"],
        ):
            arr = np.array(data, dtype=float)
            # Keep ONLY boxes within 0..50px of this edge; drop farther ones
            within_threshold = arr[(arr >= 0) & (arr <= DISTANCE_THRESHOLD)]
            ax.hist(within_threshold, bins=bins, color="#ff7f0e", edgecolor="black")
            ax.set_title(f"{title} edge distance (<={DISTANCE_THRESHOLD}px only)")
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_class_dir, "separate_edge_distance_hist.png"), dpi=150)
        plt.tight_layout()
        plt.close()

        # Additional single-plot: histogram of minimum distance to any edge
        min_d = [min(l, r, t, b) for l, r, t, b in zip(left_d, right_d, top_d, bottom_d)]
        if len(min_d) > 0:
            plt.figure(figsize=(7, 4))
            arr = np.array(min_d, dtype=float)
            within_threshold = arr[(arr >= 0) & (arr <= DISTANCE_THRESHOLD)]
            plt.hist(within_threshold, bins=bins, color="#2ca02c", edgecolor="black")
            plt.title(f"Minimum edge distance (<= {DISTANCE_THRESHOLD}px only)")
            plt.xlabel("Pixels")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(output_class_dir, "edge_distance_hist.png"), dpi=150)
            plt.tight_layout()
            plt.close()


def percentile_thresholds(areas: List[float]) -> Tuple[float, float, float]:
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError("numpy is required to compute percentiles") from e
    if len(areas) == 0:
        return 0.0, 0.0, 0.0
    p25 = float(np.percentile(areas, 25))
    p50 = float(np.percentile(areas, 50))
    p75 = float(np.percentile(areas, 75))
    return p25, p50, p75


def is_centered_within_margin(
    bbox: List[float], img_w: int, img_h: int, margin_ratio: float = 0.10
) -> bool:
    cx, cy = compute_bbox_center(bbox)
    dx = abs(cx - img_w / 2.0)
    dy = abs(cy - img_h / 2.0)
    return (dx <= margin_ratio * img_w) and (dy <= margin_ratio * img_h)


def min_edge_distance(bbox: List[float], img_w: int, img_h: int) -> float:
    l, r, t, b = distances_to_edges(bbox, img_w, img_h)
    return min(l, r, t, b)


def filter_and_rank_annotations(
    idx: CocoIndex,
    area_range: str,
    centered: Optional[bool],
    avoid_sides_px: Optional[int],
) -> List[Dict[str, Any]]:
    # Prepare area thresholds
    areas = [float(a.get("area", 0.0)) for a in idx.annotations]
    p25, p50, p75 = percentile_thresholds(areas)

    def area_in_range(a: float) -> bool:
        if area_range == "large":
            return a >= p75
        if area_range == "medium":
            # Between 25 and 50 percentile inclusive
            return (a >= p25) and (a <= p50)
        # small
        return a <= p25

    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for ann in idx.annotations:
        a = float(ann.get("area", 0.0))
        if not area_in_range(a):
            continue
        img = idx.images_by_id.get(ann["image_id"])
        if img is None:
            continue
        w = int(img.get("width", 0))
        h = int(img.get("height", 0))

        # centeredness filter if specified
        if centered is not None:
            inside = is_centered_within_margin(ann["bbox"], w, h, margin_ratio=0.10)
            if centered and not inside:
                continue
            if (not centered) and inside:
                continue

        # avoid sides filter if specified
        if avoid_sides_px is not None and avoid_sides_px != 0:
            d = min_edge_distance(ann["bbox"], w, h)
            if avoid_sides_px > 0:
                # keep only annotations at least this far from any edge
                if d < avoid_sides_px:
                    continue
            else:
                # negative: keep only annotations within |px| to some edge
                if d > abs(avoid_sides_px):
                    continue

        # Rank: primary by area (depending on range), then centeredness distance, then edge margin
        cxp, cyp, center_norm = centeredness_percent(ann["bbox"], w, h)
        center_distance_score = 1.0 - center_norm  # lower is more centered
        edge_margin = min_edge_distance(ann["bbox"], w, h)

        if area_range == "large":
            primary = -a  # larger first
        elif area_range == "small":
            primary = a  # smaller first
        else:
            # medium: prefer closer to median between p25 and p50
            target = (p25 + p50) / 2.0
            primary = abs(a - target)

        # centered preference ordering
        if centered is True:
            secondary = center_distance_score  # smaller is better
        elif centered is False:
            secondary = -center_distance_score  # larger is better
        else:
            secondary = 0.0

        # edge preference ordering
        if avoid_sides_px is None or avoid_sides_px == 0:
            tertiary = 0.0
        elif avoid_sides_px > 0:
            tertiary = -edge_margin  # prefer larger margin => more negative
        else:
            tertiary = edge_margin  # prefer smaller margin

        score_tuple = (primary, secondary, tertiary)
        # we sort ascending, so lower tuple is preferred
        total_order_key = primary * 1e6 + secondary * 1e3 + tertiary
        candidates.append((total_order_key, ann))

    # sort by combined score ascending
    candidates.sort(key=lambda x: x[0])
    return [ann for _, ann in candidates]


def build_single_annotation_coco(
    idx: CocoIndex,
    category: Dict[str, Any],
    ann: Dict[str, Any],
) -> Dict[str, Any]:
    img = idx.images_by_id[ann["image_id"]]
    return {
        "info": idx.info or {},
        "licenses": idx.licenses or [],
        "images": [img],
        "annotations": [ann],
        "categories": [category],
    }


def get_single_category(coco: Dict[str, Any]) -> Dict[str, Any]:
    categories = coco.get("categories", [])
    if len(categories) == 0:
        raise ValueError("No categories found in input JSON.")
    if len(categories) > 1:
        # Input should be per-class; fall back to first if multiple
        pass
    return categories[0]


def run_analysis_mode(input_json: str, output_root_dir: Optional[str]) -> None:
    coco = read_json(input_json)
    category = get_single_category(coco)
    class_name = sanitize_class_name(category.get("name", "class"))
    base_dir = output_root_dir or os.path.dirname(input_json)
    class_dir = os.path.join(base_dir)
    ensure_dir(class_dir)
    plot_analysis(input_json, class_dir)
    print(f"Saved analysis plots to: {class_dir}")


def run_create_mode(
    input_json: str,
    n: int,
    area_range: str,
    centered: Optional[bool],
    avoid_sides_px: Optional[int],
    output_root_dir: Optional[str],
) -> None:
    coco = read_json(input_json)
    idx = build_indices(coco)
    category = get_single_category(coco)
    class_name = sanitize_class_name(category.get("name", "class"))
    split = infer_split_from_filename(input_json)

    base_dir = output_root_dir or os.path.dirname(input_json)
    class_dir = os.path.join(base_dir, 'instances')
    ensure_dir(class_dir)

    # CSV tracking
    csv_fieldnames = ["id", "area", "centered", "avoid_sides"]
    csv_path = os.path.join(class_dir, "ref_image_selections.csv")
    ensure_csv_with_header(csv_path, csv_fieldnames)
    existing_by_id = read_csv_records_by_id(csv_path)

    ranked = filter_and_rank_annotations(
        idx=idx,
        area_range=area_range,
        centered=centered,
        avoid_sides_px=avoid_sides_px,
    )
    if len(ranked) == 0:
        print("No annotations match the given criteria.")
        return

    # Build up to N selections, skipping IDs already present in CSV
    selections: List[Dict[str, Any]] = []
    attempted_count = 0
    for ann in ranked:
        if len(selections) >= n:
            break
        attempted_count += 1
        ann_id_str = str(ann["id"])
        if ann_id_str in existing_by_id:
            # Warn about duplicate attempt with details
            existing = existing_by_id[ann_id_str]
            attempted = {
                "id": ann_id_str,
                "area": area_range,
                "centered": ("true" if centered is True else "false") if centered is not None else "",
                "avoid_sides": "" if avoid_sides_px is None else str(avoid_sides_px),
            }
            yellow_warning(
                f"Duplicate selection skipped for id={ann_id_str}. "
                f"Existing entry: {existing}. Attempted: {attempted}."
            )
            continue
        selections.append(ann)

    if len(selections) == 0:
        print("All candidate annotation IDs are already recorded in CSV. Nothing to create.")
        return

    for ann in selections:
        out_name = f"{class_name}_{ann['id']}_instances_{split}.json"
        out_path = os.path.join(class_dir, out_name)
        single_coco = build_single_annotation_coco(idx, category, ann)
        write_json(out_path, single_coco)
        print(f"Wrote: {out_path} (img_id={ann['image_id']}, ann_id={ann['id']})")

        # Append record to CSV
        record = {
            "id": str(ann["id"]),
            "area": area_range,
            "centered": ("true" if centered is True else "false") if centered is not None else "",
            "avoid_sides": "" if avoid_sides_px is None else str(avoid_sides_px),
        }
        append_csv_record(csv_path, csv_fieldnames, record)
        # Update in-memory map to avoid duplicates if loop continues for large N
        existing_by_id[str(ann["id"])] = record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze and select per-annotation reference JSONs for a single COCO class.\n"
            "- mode=analysis: save area histogram, 2D centeredness hist, edge distance hists.\n"
            "- mode=create: create N single-annotation JSONs using heuristics."
        )
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to per-class instances JSON (e.g., data/coco/annotations/per_class_instances/dog_instances_train2017.json)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["analysis", "create"],
        required=True,
        help="Run in analysis or create mode.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root dir to place the <class>/ outputs. Defaults to the input JSON directory.",
    )
    # Create-mode arguments
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of per-annotation JSONs to create in create mode.",
    )
    parser.add_argument(
        "--area_range",
        type=str,
        choices=["large", "medium", "small"],
        default="large",
        help="Area range: large (>=75p), medium (25-50p), small (<=25p).",
    )
    parser.add_argument(
        "--centered",
        type=parse_bool,
        default=None,
        help="True: only centered within 10%% margin; False: exclude centered; None: ignore.",
    )
    parser.add_argument(
        "--avoid-sides",
        dest="avoid_sides",
        type=int,
        default=None,
        help="Positive px: avoid edges by at least px. Negative: select within |px| of an edge. None: ignore.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "analysis":
        run_analysis_mode(
            input_json=args.input_json,
            output_root_dir=args.output_root,
        )
    else:
        run_create_mode(
            input_json=args.input_json,
            n=args.n,
            area_range=args.area_range,
            centered=args.centered,
            avoid_sides_px=args.avoid_sides,
            output_root_dir=args.output_root,
        )


if __name__ == "__main__":
    main()


