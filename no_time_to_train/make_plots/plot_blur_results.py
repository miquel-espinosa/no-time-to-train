import argparse
import os
import re
from typing import List, Tuple, Optional

from PIL import Image
import matplotlib.pyplot as plt


"""
Usage:

python no_time_to_train/make_plots/plot_blur_results.py \
    --results-root ./work_dirs/blur_ablation \
    --class-id 0 \
    --max-cols 4 \
    --output-dir ./no_time_to_train/make_plots/blur_ablation \
    --outfile grid_blur_ablation_class_0.png
"""

KEEP_BLUR_LEVELS = [0, 7, 11, 15, 21, 31, 41, 51, 61, 71, 81, 91]

def list_blur_dirs(results_root: str) -> List[Tuple[int, str]]:
	"""
	Find subdirectories named 'blur_<level>' and return a sorted list by <level>.
	"""
	entries = []
	if not os.path.isdir(results_root):
		return entries
	for name in os.listdir(results_root):
		full_path = os.path.join(results_root, name)
		if not os.path.isdir(full_path):
			continue
		m = re.fullmatch(r"blur_(\d+)", name)
		if m:
			level = int(m.group(1))
			if level in KEEP_BLUR_LEVELS:
				entries.append((level, full_path))
	entries.sort(key=lambda x: x[0])
	return entries


def parse_eval_stats(stats_path: str) -> Tuple[Optional[float], Optional[float]]:
	"""
	Parse the first AP (IoU=0.50:0.95) from the BBOX and SEGM sections.
	Returns (bbox_ap, segm_ap). Returns (None, None) if not found.
	"""
	if not os.path.isfile(stats_path):
		return None, None

	with open(stats_path, "r", encoding="utf-8", errors="ignore") as f:
		lines = [line.strip() for line in f.readlines()]

	def extract_first_ap(start_idx: int) -> Optional[float]:
		for i in range(start_idx + 1, min(start_idx + 6, len(lines))):
			# Expecting: "AP IoU=0.50:0.95: 0.4782"
			m = re.match(r"AP IoU=0\.50:0\.95:\s*([0-9]*\.?[0-9]+)", lines[i])
			if m:
				return float(m.group(1))
		return None

	bbox_ap = None
	segm_ap = None

	for idx, line in enumerate(lines):
		if bbox_ap is None and "===== BBOX RESULTS" in line:
			bbox_ap = extract_first_ap(idx)
		if segm_ap is None and "===== SEGM RESULTS" in line:
			segm_ap = extract_first_ap(idx)
		if bbox_ap is not None and segm_ap is not None:
			break

	return bbox_ap, segm_ap


def find_memory_vis_image(blur_dir: str, class_id: int) -> Optional[str]:
	"""
	Find first image in memory_vis starting with '<class_id>_'.
	"""
	memory_vis_dir = os.path.join(blur_dir, "memory_vis")
	if not os.path.isdir(memory_vis_dir):
		return None
	prefix = f"{class_id}_"
	candidates = [f for f in os.listdir(memory_vis_dir) if f.startswith(prefix) and f.lower().endswith(".png")]
	if not candidates:
		return None
	candidates.sort()
	return os.path.join(memory_vis_dir, candidates[0])


def crop_ori_and_pca(composite_img: Image.Image, margin_width_in_composite: int = 5) -> Image.Image:
	"""
	The composite is [ori | margin(white) | kmeans | margin(white) | pca] along width.
	We crop ori and pca, and re-concatenate them with a black margin between.
	"""
	width_total, height_total = composite_img.size
	# width_total = 3W + 2*margin
	# assume two margins of equal width (default 5 px); derive W
	assumed_margin_total = 2 * margin_width_in_composite
	if width_total <= assumed_margin_total:
		return composite_img.copy()
	unit_w = (width_total - assumed_margin_total) // 3

	# Crop original
	ori_left = 0
	ori_right = unit_w
	ori_img = composite_img.crop((ori_left, 0, ori_right, height_total))

	# Crop PCA (last unit)
	pca_left = 2 * unit_w + 2 * margin_width_in_composite
	pca_right = pca_left + unit_w
	pca_img = composite_img.crop((pca_left, 0, pca_right, height_total))

	# Create black margin between both (use same width as original white margin)
	black_margin = Image.new("RGB", (margin_width_in_composite, height_total), color=(0, 0, 0))

	# Concatenate: ori | black | pca
	out_width = ori_img.width + black_margin.width + pca_img.width
	out = Image.new("RGB", (out_width, height_total))
	x = 0
	out.paste(ori_img, (x, 0))
	x += ori_img.width
	out.paste(black_margin, (x, 0))
	x += black_margin.width
	out.paste(pca_img, (x, 0))
	return out


def build_grid(
	per_blur_images: List[Tuple[int, Image.Image, Optional[float], Optional[float]]],
	max_cols: int = 3
) -> plt.Figure:
	"""
	Create a matplotlib grid where each cell contains the concatenated (ori|pca) image.
	Top title: "Blur <level>"
	Bottom caption (under image): "bbox AP=<>, segm AP=<>"
	"""
	num_items = len(per_blur_images)
	if num_items == 0:
		raise ValueError("No images to plot.")

	cols = max(1, min(max_cols, num_items))
	rows = (num_items + cols - 1) // cols

	# Use a smaller, tighter layout (reduce per-row height since images are wide)
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.2), squeeze=False, constrained_layout=False)

	for idx, (blur_level, img, bbox_ap, segm_ap) in enumerate(per_blur_images):
		r = idx // cols
		c = idx % cols
		ax = axes[r][c]
		ax.imshow(img)
		ax.axis("off")

		# Top title: blur level
		ax.set_title(f"Gaussian Blur (ksize={blur_level})", fontsize=11, pad=4)

		# Bottom caption: bbox/segm AP underneath the image
		if bbox_ap is not None:
			bbox_str = f"bbox AP={bbox_ap:.4f}"
		else:
			bbox_str = "bbox AP=N/A"
		if segm_ap is not None:
			segm_str = f"segm AP={segm_ap:.4f}"
		else:
			segm_str = "segm AP=N/A"
		ax.text(
			0.5,
			-0.05,
			f"{bbox_str} | {segm_str}",
			transform=ax.transAxes,
			ha="center",
			va="top",
			fontsize=9
		)

	# Hide any extra axes
	for idx in range(num_items, rows * cols):
		r = idx // cols
		c = idx % cols
		axes[r][c].axis("off")

	# Make grid compact: small spaces and accommodate text below images
	fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0.03, right=0.97, top=0.95, bottom=0.06)

	return fig


def main() -> None:
	parser = argparse.ArgumentParser(description="Create a grid of blur ablation results (ori + DINO PCA).")
	parser.add_argument(
		"--results-root",
		type=str,
		default=".",
		help="Directory containing 'blur_*' folders (default: current directory)."
	)
	parser.add_argument(
		"--class-id",
		"-c",
		type=int,
		required=True,
		help="Class id to select from memory_vis (e.g., 0 selects files starting with '0_')."
	)
	parser.add_argument(
		"--max-cols",
		type=int,
		default=3,
		help="Maximum number of columns in the grid (default: 3)."
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=os.path.abspath(os.path.join(os.path.dirname(__file__), "blur_ablation")),
		help="Directory to save the final grid image."
	)
	parser.add_argument(
		"--outfile",
		type=str,
		default=None,
		help="Optional output filename. If not provided, a default using class id is used."
	)
	args = parser.parse_args()

	blur_dirs = list_blur_dirs(args.results_root)
	if not blur_dirs:
		raise SystemExit(f"No 'blur_*' directories found under: {args.results_root}")

	per_blur: List[Tuple[int, Image.Image, Optional[float], Optional[float]]] = []
	for blur_level, blur_dir in blur_dirs:
		stats_path = os.path.join(blur_dir, "coco_eval_stats_.txt")
		bbox_ap, segm_ap = parse_eval_stats(stats_path)

		mem_img_path = find_memory_vis_image(blur_dir, args.class_id)
		if mem_img_path is None or not os.path.isfile(mem_img_path):
			# Skip if the memory_vis image for this class id is not present
			continue

		composite = Image.open(mem_img_path).convert("RGB")
		ori_plus_pca = crop_ori_and_pca(composite, margin_width_in_composite=5)
		per_blur.append((blur_level, ori_plus_pca, bbox_ap, segm_ap))

	# Ensure ordering by blur level
	per_blur.sort(key=lambda x: x[0])
	if not per_blur:
		raise SystemExit(f"No memory_vis images found for class id {args.class_id} in the discovered blur folders.")

	fig = build_grid(per_blur, max_cols=args.max_cols)

	os.makedirs(args.output_dir, exist_ok=True)
	outfile = args.outfile
	if not outfile:
		outfile = f"grid_blur_ablation_class_{args.class_id}.png"
	out_path = os.path.join(args.output_dir, outfile)
	fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)
	print(f"Saved grid to: {out_path}")


if __name__ == "__main__":
	main()

