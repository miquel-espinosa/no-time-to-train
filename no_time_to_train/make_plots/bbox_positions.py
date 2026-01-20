import argparse
import os
from typing import List, Tuple, Optional

from PIL import Image
import matplotlib.pyplot as plt

from no_time_to_train.dataset.metainfo import METAINFO


"""
Usage:

python no_time_to_train/make_plots/bbox_positions.py \
	--per-class-root data/coco/annotations/per_class_instances \
	--filename centeredness_2d_hist_plain.png \
	--max-cols 6 \
	--output-dir ./no_time_to_train/make_plots/bbox_positions \
	--outfile grid_bbox_positions.png
"""

# Only use few-shot classes as requested
COCO_CLASSES: List[str] = list(METAINFO["few_shot_classes"])


def find_image_for_class(per_class_root: str, class_name: str, filename: str) -> Optional[str]:
	"""
	Return the absolute path to the requested PNG for the given class, if it exists.
	"""
	class_dir = os.path.join(per_class_root, class_name)
	image_path = os.path.join(class_dir, filename)
	if os.path.isfile(image_path):
		return os.path.abspath(image_path)
	return None


def build_grid(per_class_images: List[Tuple[str, Image.Image]], max_cols: int = 6) -> plt.Figure:
	"""
	Create a matplotlib grid where each cell contains the class image.
	Each subplot title is the class name (only).
	"""
	num_items = len(per_class_images)
	if num_items == 0:
		raise ValueError("No images to plot.")

	cols = max(1, min(max_cols, num_items))
	rows = (num_items + cols - 1) // cols

	# Tight layout to accommodate titles while keeping images readable
	fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8), squeeze=False, constrained_layout=False)

	for idx, (class_name, img) in enumerate(per_class_images):
		r = idx // cols
		c = idx % cols
		ax = axes[r][c]
		ax.imshow(img)
		ax.axis("off")
		ax.set_title(class_name, fontsize=18, pad=4)

	# Hide any extra axes
	for idx in range(num_items, rows * cols):
		r = idx // cols
		c = idx % cols
		axes[r][c].axis("off")

	# Compact spacing
	fig.subplots_adjust(wspace=-0.5, hspace=0.18, left=0.03, right=0.97, top=0.95, bottom=0.06)
	return fig


def main() -> None:
	parser = argparse.ArgumentParser(description="Create a grid of bbox position plots for COCO few-shot classes.")
	parser.add_argument(
		"--per-class-root",
		type=str,
		default="data/coco/annotations/per_class_instances",
		help="Directory containing per-class subfolders (default: data/coco/annotations/per_class_instances)."
	)
	parser.add_argument(
		"--filename",
		type=str,
		default="centeredness_2d_hist_plain.png",
		help="Filename of the precomputed image inside each class folder (default: centeredness_2d_hist_plain.png)."
	)
	parser.add_argument(
		"--max-cols",
		type=int,
		default=6,
		help="Maximum number of columns in the grid (default: 6)."
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=os.path.abspath(os.path.join(os.path.dirname(__file__), "bbox_positions")),
		help="Directory to save the final grid image."
	)
	parser.add_argument(
		"--outfile",
		type=str,
		default=None,
		help="Optional output filename. If not provided, a default is used."
	)
	args = parser.parse_args()

	available: List[Tuple[str, Image.Image]] = []
	for class_name in COCO_CLASSES:
		img_path = find_image_for_class(args.per_class_root, class_name, args.filename)
		if img_path is None:
			continue
		img = Image.open(img_path).convert("RGB")
		available.append((class_name, img))

	# Sort by class name for consistent ordering
	available.sort(key=lambda x: x[0])
	if not available:
		raise SystemExit(
			f"No images found for the selected classes under: {args.per_class_root} (filename='{args.filename}')."
		)

	fig = build_grid(available, max_cols=args.max_cols)

	os.makedirs(args.output_dir, exist_ok=True)
	outfile = args.outfile or "grid_bbox_positions.png"
	out_path = os.path.join(args.output_dir, outfile)
	plt.tight_layout()
	fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
	plt.close(fig)
	print(f"Saved grid to: {out_path}")


if __name__ == "__main__":
	main()


