import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from no_time_to_train.dataset.metainfo import METAINFO

COCO_CLASSES = METAINFO["few_shot_classes"]

"""
python no_time_to_train/make_plots/mask_area_distribution.py \
  --input data/coco/annotations/instances_val2017.json \
  --output no_time_to_train/make_plots/mask_area_distribution/mask_area_distribution.png \
  --edges-output no_time_to_train/make_plots/mask_area_distribution/bbox_edge_distance_histograms.png \
  --center-output no_time_to_train/make_plots/mask_area_distribution/bbox_center_density.png \
  --bins 80 \
  --distance-bins 80 \
  --disable-center-density
"""

def resolve_default_input_path(script_path: Path) -> Path:
	"""
	Resolve the default COCO annotations path relative to the repository root.
	Assumes this file is located at: <repo_root>/no_time_to_train/make_plots/...
	and the COCO file is at:        <repo_root>/data/coco/annotations/instances_val2017.json
	"""
	repo_root = script_path.resolve().parents[2]
	return repo_root / "data" / "coco" / "annotations" / "instances_val2017.json"


def get_few_shot_category_ids(coco_annotations_path: Path) -> Set[int]:
	"""
	Load COCO annotations and return the set of category IDs that correspond to few-shot classes.
	"""
	if not coco_annotations_path.exists():
		raise FileNotFoundError(f"COCO annotations not found at: {coco_annotations_path}")

	with coco_annotations_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	categories = data.get("categories", [])
	few_shot_class_names = set(COCO_CLASSES)
	category_ids = set()

	for cat in categories:
		if cat.get("name") in few_shot_class_names:
			category_ids.add(cat["id"])

	return category_ids


def load_coco_mask_areas(coco_annotations_path: Path) -> List[float]:
	"""
	Load COCO annotations and return a list of mask areas (in pixels).
	Uses the 'area' field from each annotation, which corresponds to the segmentation area.
	Only includes annotations from few-shot classes.
	"""
	if not coco_annotations_path.exists():
		raise FileNotFoundError(f"COCO annotations not found at: {coco_annotations_path}")

	few_shot_category_ids = get_few_shot_category_ids(coco_annotations_path)

	with coco_annotations_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	annotations = data.get("annotations", [])
	areas: List[float] = []

	for ann in annotations:
		# Filter by category
		category_id = ann.get("category_id", None)
		if category_id not in few_shot_category_ids:
			continue

		area = ann.get("area", None)
		if area is None:
			continue
		# Keep only positive areas
		if isinstance(area, (int, float)) and area > 0:
			areas.append(float(area))

	return areas


def plot_area_distribution(
	areas: List[float],
	output_path: Path,
	title_suffix: str = "",
	clip_quantile: float | None = 0.99,
	bins: int | str = "auto",
	log_x: bool = False,
	log_y: bool = True,
	kde: bool = False,
) -> None:
	"""
	Create a single histogram + KDE with optional log scaling on x and/or y axes.
	
	Args:
		areas: List of mask areas in pixels
		output_path: Where to save the plot
		title_suffix: Optional suffix for the title
		clip_quantile: Optional quantile to clip outliers
		bins: Number of bins or binning strategy
		log_x: If True, plot x-axis (area) on log scale
		log_y: If True, plot y-axis (count) on log scale
		kde: If True, overlay a KDE curve on the histogram
	"""
	if len(areas) == 0:
		raise ValueError("No mask areas found to plot.")

	areas_array = np.asarray(areas, dtype=float)
	# Only positive areas are kept earlier; safeguard small epsilons
	areas_array = areas_array[areas_array > 0]
	
	# Determine what data to plot based on log_x
	if log_x:
		plot_data = np.log10(areas_array)
	else:
		plot_data = areas_array

	# Seaborn aesthetics
	sns.set_theme(style="whitegrid", context="talk")

	fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
	sns.histplot(plot_data, bins=bins, kde=kde, ax=ax, color="#2563eb", edgecolor="white")
	
	# Build title based on log settings
	scale_desc = []
	if log_x:
		scale_desc.append("log x")
	if log_y:
		scale_desc.append("log y")
	
	# Set x-axis label based on log_x
	if log_x:
		ax.set_xlabel("Mask area (px^2, log10)")
	else:
		ax.set_xlabel("Mask area (px^2)")
	
	# Set y-axis label and scale based on log_y
	if log_y:
		ax.set_ylabel("Count (log)")
		ax.set_yscale("log")
	else:
		ax.set_ylabel("Count")

	n = len(areas_array)
	fig.suptitle(f"Mask area distribution", fontsize=18)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=200)
	plt.close(fig)


def load_coco_bbox_distances_and_centers(coco_annotations_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Load COCO annotations and compute normalized distances from bbox edges to the image frame,
	as well as normalized bbox center coordinates.
	Only includes annotations from few-shot classes.
	Returns arrays in order: d_left, d_top, d_right, d_bottom, c_x, c_y.
	"""
	if not coco_annotations_path.exists():
		raise FileNotFoundError(f"COCO annotations not found at: {coco_annotations_path}")

	few_shot_category_ids = get_few_shot_category_ids(coco_annotations_path)

	with coco_annotations_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	# Map image_id -> (W, H)
	images: List[Dict] = data.get("images", [])
	image_id_to_size: Dict[int, Tuple[int, int]] = {img["id"]: (img["width"], img["height"]) for img in images if "id" in img and "width" in img and "height" in img}

	d_left: List[float] = []
	d_top: List[float] = []
	d_right: List[float] = []
	d_bottom: List[float] = []
	cx: List[float] = []
	cy: List[float] = []

	for ann in data.get("annotations", []):
		# Filter by category
		category_id = ann.get("category_id", None)
		if category_id not in few_shot_category_ids:
			continue

		bbox = ann.get("bbox", None)
		image_id = ann.get("image_id", None)
		if bbox is None or image_id is None:
			continue
		if image_id not in image_id_to_size:
			continue
		W, H = image_id_to_size[image_id]
		if W <= 0 or H <= 0:
			continue
		# COCO bbox format: [x, y, w, h]
		try:
			x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
		except Exception:
			continue
		if w <= 0 or h <= 0:
			continue

		# Distances to edges, normalized to [0, 1] by W/H
		left = x / W
		top = y / H
		right = (W - (x + w)) / W
		bottom = (H - (y + h)) / H

		# Clamp to [0, 1] to avoid minor numerical issues from annotations
		left = float(np.clip(left, 0.0, 1.0))
		top = float(np.clip(top, 0.0, 1.0))
		right = float(np.clip(right, 0.0, 1.0))
		bottom = float(np.clip(bottom, 0.0, 1.0))

		d_left.append(left)
		d_top.append(top)
		d_right.append(right)
		d_bottom.append(bottom)

		# Normalized center coordinates
		center_x = (x + 0.5 * w) / W
		center_y = (y + 0.5 * h) / H
		cx.append(float(np.clip(center_x, 0.0, 1.0)))
		cy.append(float(np.clip(center_y, 0.0, 1.0)))

	return (
		np.asarray(d_left, dtype=float),
		np.asarray(d_top, dtype=float),
		np.asarray(d_right, dtype=float),
		np.asarray(d_bottom, dtype=float),
		np.asarray(cx, dtype=float),
		np.asarray(cy, dtype=float),
	)


def load_coco_bbox_distances(coco_annotations_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Load COCO annotations and compute normalized distances from bbox edges to the image frame.
	Only includes annotations from few-shot classes.
	Returns arrays in order: d_left, d_top, d_right, d_bottom.
	"""
	if not coco_annotations_path.exists():
		raise FileNotFoundError(f"COCO annotations not found at: {coco_annotations_path}")

	few_shot_category_ids = get_few_shot_category_ids(coco_annotations_path)

	with coco_annotations_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	images: List[Dict] = data.get("images", [])
	image_id_to_size: Dict[int, Tuple[int, int]] = {img["id"]: (img["width"], img["height"]) for img in images if "id" in img and "width" in img and "height" in img}

	d_left: List[float] = []
	d_top: List[float] = []
	d_right: List[float] = []
	d_bottom: List[float] = []

	for ann in data.get("annotations", []):
		# Filter by category
		category_id = ann.get("category_id", None)
		if category_id not in few_shot_category_ids:
			continue

		bbox = ann.get("bbox", None)
		image_id = ann.get("image_id", None)
		if bbox is None or image_id is None:
			continue
		if image_id not in image_id_to_size:
			continue
		W, H = image_id_to_size[image_id]
		if W <= 0 or H <= 0:
			continue
		try:
			x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
		except Exception:
			continue
		if w <= 0 or h <= 0:
			continue

		left = float(np.clip(x / W, 0.0, 1.0))
		top = float(np.clip(y / H, 0.0, 1.0))
		right = float(np.clip((W - (x + w)) / W, 0.0, 1.0))
		bottom = float(np.clip((H - (y + h)) / H, 0.0, 1.0))

		d_left.append(left)
		d_top.append(top)
		d_right.append(right)
		d_bottom.append(bottom)

	return (
		np.asarray(d_left, dtype=float),
		np.asarray(d_top, dtype=float),
		np.asarray(d_right, dtype=float),
		np.asarray(d_bottom, dtype=float),
	)


def load_coco_bbox_centers(coco_annotations_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Load COCO annotations and compute normalized bbox center coordinates.
	Only includes annotations from few-shot classes.
	Returns arrays in order: c_x, c_y.
	"""
	if not coco_annotations_path.exists():
		raise FileNotFoundError(f"COCO annotations not found at: {coco_annotations_path}")

	few_shot_category_ids = get_few_shot_category_ids(coco_annotations_path)

	with coco_annotations_path.open("r", encoding="utf-8") as f:
		data = json.load(f)

	images: List[Dict] = data.get("images", [])
	image_id_to_size: Dict[int, Tuple[int, int]] = {img["id"]: (img["width"], img["height"]) for img in images if "id" in img and "width" in img and "height" in img}

	cx: List[float] = []
	cy: List[float] = []

	for ann in data.get("annotations", []):
		# Filter by category
		category_id = ann.get("category_id", None)
		if category_id not in few_shot_category_ids:
			continue

		bbox = ann.get("bbox", None)
		image_id = ann.get("image_id", None)
		if bbox is None or image_id is None:
			continue
		if image_id not in image_id_to_size:
			continue
		W, H = image_id_to_size[image_id]
		if W <= 0 or H <= 0:
			continue
		try:
			x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
		except Exception:
			continue
		if w <= 0 or h <= 0:
			continue

		center_x = float(np.clip((x + 0.5 * w) / W, 0.0, 1.0))
		center_y = float(np.clip((y + 0.5 * h) / H, 0.0, 1.0))
		cx.append(center_x)
		cy.append(center_y)

	return (
		np.asarray(cx, dtype=float),
		np.asarray(cy, dtype=float),
	)

def plot_bbox_edge_distance_histograms(
	d_left: np.ndarray,
	d_top: np.ndarray,
	d_right: np.ndarray,
	d_bottom: np.ndarray,
	output_path: Path,
	bins: int = 80,
) -> None:
	"""
	Create a 2x2 grid of histograms (with KDE) for distances of bbox edges to image frame.
	Each axis shows normalized distance in [0, 1].
	"""
	if min(len(d_left), len(d_top), len(d_right), len(d_bottom)) == 0:
		raise ValueError("No bbox edge distances available to plot.")

	sns.set_theme(style="whitegrid", context="talk")
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), constrained_layout=True, sharey='row')
	cfg = dict(bins=bins, kde=True, edgecolor="white")

	sns.histplot(d_left, color="#1f77b4", ax=axes[0, 0], **cfg)
	axes[0, 0].set_title("Normalized distance to left image frame")
	axes[0, 0].set_xlim(0, 1)
	axes[0, 0].set_ylabel("Count")

	sns.histplot(d_right, color="#ff7f0e", ax=axes[0, 1], **cfg)
	axes[0, 1].set_title("Normalized distance to right image frame")
	axes[0, 1].set_xlim(0, 1)
	axes[0, 1].set_ylabel("")

	sns.histplot(d_top, color="#2ca02c", ax=axes[1, 0], **cfg)
	axes[1, 0].set_title("Normalized distance to top image frame")
	axes[1, 0].set_xlim(0, 1)
	axes[1, 0].set_ylabel("Count")

	sns.histplot(d_bottom, color="#d62728", ax=axes[1, 1], **cfg)
	axes[1, 1].set_title("Normalized distance to bottom image frame")
	axes[1, 1].set_xlim(0, 1)
	axes[1, 1].set_ylabel("")

	fig.suptitle("BBOX distances to image frame", fontsize=18)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=300)
	plt.close(fig)


def plot_bbox_center_density(
	center_x: np.ndarray,
	center_y: np.ndarray,
	output_path: Path,
	sample_for_scatter: int | None = 20000,
) -> None:
	"""
	Create a 2D density plot (KDE) of normalized bbox centers in [0, 1] x [0, 1].
	Optionally overlays a light scatter of a random sample for visual texture.
	"""
	if len(center_x) == 0 or len(center_y) == 0:
		raise ValueError("No bbox centers available to plot.")

	sns.set_theme(style="white", context="talk")
	fig, ax = plt.subplots(figsize=(7.5, 7), constrained_layout=True)

	# 2D KDE heatmap
	sns.kdeplot(
		x=center_x,
		y=center_y,
		fill=True,
		levels=50,
		thresh=0.05,
		cmap="viridis",
		ax=ax,
	)

	# Optional light scatter overlay for point texture
	if sample_for_scatter is not None and sample_for_scatter > 0:
		n = len(center_x)
		if n > sample_for_scatter:
			idx = np.random.choice(n, size=sample_for_scatter, replace=False)
			xs = center_x[idx]
			ys = center_y[idx]
		else:
			xs = center_x
			ys = center_y
		ax.scatter(xs, ys, s=3, c="white", alpha=0.08, edgecolors="none")

	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)
	ax.set_xlabel("Normalized center x")
	ax.set_ylabel("Normalized center y")
	ax.set_title("COCO val2017 — Bbox center position density (few-shot classes)")
	ax.set_aspect("equal", adjustable="box")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=220)
	plt.close(fig)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="COCO val2017 plots: mask area distribution (log), bbox edge distance histograms, and optional bbox center density."
	)
	default_input = resolve_default_input_path(Path(__file__))
	default_output = Path(__file__).resolve().parent / "mask_area_distribution.png"

	parser.add_argument(
		"--input",
		type=Path,
		default=default_input,
		help=f"Path to COCO annotations JSON (default: {default_input})",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=default_output,
		help=f"Output image path (default: {default_output})",
	)
	parser.add_argument(
		"--bins",
		type=int,
		default=80,
		help="Number of log-scale bins for mask area histogram (default: 80). Use 'auto' by passing 0.",
	)
	parser.add_argument(
		"--no-clip",
		action="store_true",
		help="Disable clipping the linear-scale subplot at the 99th percentile.",
	)
	parser.add_argument(
		"--clip-quantile",
		type=float,
		default=0.99,
		help="Clip quantile for the linear-scale subplot (default: 0.99). Ignored if --no-clip.",
	)
	# Bbox edge distances histograms
	default_edges_output = Path(__file__).resolve().parent / "bbox_edge_distance_histograms.png"
	parser.add_argument(
		"--edges-output",
		type=Path,
		default=default_edges_output,
		help=f"Output path for bbox edge distance histograms (default: {default_edges_output})",
	)
	parser.add_argument(
		"--distance-bins",
		type=int,
		default=80,
		help="Number of bins for bbox edge distance histograms (default: 80).",
	)
	parser.add_argument(
		"--disable-bbox-hists",
		action="store_true",
		help="Skip generating bbox edge distance histograms.",
	)
	# Bbox center density
	default_center_output = Path(__file__).resolve().parent / "bbox_center_density.png"
	parser.add_argument(
		"--center-output",
		type=Path,
		default=default_center_output,
		help=f"Output path for bbox center density plot (default: {default_center_output})",
	)
	parser.add_argument(
		"--disable-center-density",
		action="store_true",
		help="Skip generating bbox center density plot and its computation.",
	)
	parser.add_argument(
		"--scatter-sample",
		type=int,
		default=20000,
		help="Number of random points to overlay as scatter on the center density plot (default: 20000). Set 0 to disable.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	bins_arg: int | str = "auto" if args.bins == 0 else args.bins
	clip_q = None if args.no_clip else args.clip_quantile

	# Ensure output directory exists
	args.output.parent.mkdir(parents=True, exist_ok=True)

	areas = load_coco_mask_areas(args.input)
	title_suffix = args.input.name
	plot_area_distribution(
		areas=areas,
		output_path=args.output,
		title_suffix=title_suffix,
		clip_quantile=clip_q,
		bins=bins_arg,
	)
	print(f"Saved plot to: {args.output}")

	if not args.disable_bbox_hists:
		args.edges_output.parent.mkdir(parents=True, exist_ok=True)
		d_left, d_top, d_right, d_bottom = load_coco_bbox_distances(args.input)
		plot_bbox_edge_distance_histograms(
			d_left=d_left,
			d_top=d_top,
			d_right=d_right,
			d_bottom=d_bottom,
			output_path=args.edges_output,
			bins=args.distance_bins,
		)
		print(f"Saved bbox edge distance histograms to: {args.edges_output}")

	if not args.disable_center_density:
		args.center_output.parent.mkdir(parents=True, exist_ok=True)
		c_x, c_y = load_coco_bbox_centers(args.input)
		sample_n = None if args.scatter_sample <= 0 else args.scatter_sample
		plot_bbox_center_density(
			center_x=c_x,
			center_y=c_y,
			output_path=args.center_output,
			sample_for_scatter=sample_n,
		)
		print(f"Saved bbox center density plot to: {args.center_output}")


if __name__ == "__main__":
	main()

