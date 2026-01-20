import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from openTSNE import TSNE
from pycocotools.coco import COCO

try:
	from sklearn.decomposition import PCA
except Exception:
	PCA = None  # type: ignore


"""
# To extract features
python no_time_to_train/make_plots/tsne-coco.py --extract

# To plot for specific classes
python no_time_to_train/make_plots/tsne-coco.py --classes hot_dog sandwich
python no_time_to_train/make_plots/tsne-coco.py --classes suitcase handbag
python no_time_to_train/make_plots/tsne-coco.py --classes truck bus
python no_time_to_train/make_plots/tsne-coco.py --classes truck bus car motorcycle bicycle
python no_time_to_train/make_plots/tsne-coco.py --classes car truck
python no_time_to_train/make_plots/tsne-coco.py --classes stop_sign traffic_light
python no_time_to_train/make_plots/tsne-coco.py --classes oven microwave
python no_time_to_train/make_plots/tsne-coco.py --classes cup wine_glass vase
python no_time_to_train/make_plots/tsne-coco.py --classes spoon fork
python no_time_to_train/make_plots/tsne-coco.py --classes chair couch
python no_time_to_train/make_plots/tsne-coco.py --classes tv laptop

python no_time_to_train/make_plots/tsne-coco.py --classes chair dining_table
python no_time_to_train/make_plots/tsne-coco.py --classes bicycle motorcycle
python no_time_to_train/make_plots/tsne-coco.py --classes airplane bird
"""

"""
COCO CLASSES:
	'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
	'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
	'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
	'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
	'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
	'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
	'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
	'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
	'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
	'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
	'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
	'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
	'scissors', 'teddy bear', 'hair drier', 'toothbrush'
"""

"""
COCO NOVEL CLASSES

	'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
	'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
	'cow', 'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv'

"""


COCO_ROOT = Path("./data/coco")
ANN_ROOT = COCO_ROOT / "annotations"
VAL_IMG_DIR = COCO_ROOT / "val2017"

DINO_MODEL = "dinov2_l"

if DINO_MODEL == "dinov2_l":
	OUT_DIR = COCO_ROOT / "dinov2_l-features"
	DINO_REPO_DIR = Path("./dinov2")
	DINO_CKPT = Path("./checkpoints/dinov2/dinov2_vitl14_pretrain.pth")
	PLOTS_DIR = Path("no_time_to_train/make_plots/tsne_plots/dinov2_l")
elif DINO_MODEL == "dinov3_l":
	OUT_DIR = COCO_ROOT / "dinov3_l-features"
	DINO_REPO_DIR = Path("./dinov3")
	DINO_CKPT = Path("./checkpoints/dinov3/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
	PLOTS_DIR = Path("no_time_to_train/make_plots/tsne_plots/dinov3_l")
elif DINO_MODEL == "dinov3_b":
	OUT_DIR = COCO_ROOT / "dinov3_b-features"
	DINO_REPO_DIR = Path("./dinov3")
	DINO_CKPT = Path("./checkpoints/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
	PLOTS_DIR = Path("no_time_to_train/make_plots/tsne_plots/dinov3_b")
elif DINO_MODEL == "dinov3_h":
	OUT_DIR = COCO_ROOT / "dinov3_h-features"
	DINO_REPO_DIR = Path("./dinov3")
	DINO_CKPT = Path("./checkpoints/dinov3/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth")
	PLOTS_DIR = Path("no_time_to_train/make_plots/tsne_plots/dinov3_h")


def find_val_annotations_json() -> Path:
	if ANN_ROOT.exists():
		candidates = list(ANN_ROOT.glob("instances_val*2017.json"))
		# Be tolerant with typos like "vale"
		candidates += list(ANN_ROOT.glob("instances_vale*2017.json"))
		if candidates:
			# Prefer the correctly spelled one if present
			for p in candidates:
				if "instances_val2017" in p.stem:
					return p
			return candidates[0]
	raise FileNotFoundError(f"COCO annotations directory not found at {ANN_ROOT}")


def ensure_out_dir() -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():  # macOS
		return torch.device("mps")
	return torch.device("cpu")


def _load_state_dict_flex(model: torch.nn.Module, ckpt_path: Path) -> None:
	ckpt = torch.load(str(ckpt_path), map_location="cpu")
	if isinstance(ckpt, dict):
		state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
	else:
		state = ckpt
	if not isinstance(state, dict):
		raise RuntimeError(f"Unexpected checkpoint format in {ckpt_path}")
	# Strip common wrappers
	new_state = {}
	for k, v in state.items():
		if k.startswith("module."):
			k = k[len("module.") :]
		new_state[k] = v
	missing, unexpected = model.load_state_dict(new_state, strict=False)
	print(f"Loaded weights from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")


def load_dinov2_model(device: torch.device):
	"""
	Load DINOv2 or DINOv3 ViT-L from the locally cloned repo and checkpoint if available.
	Fallbacks to hub if needed.
	"""
	model = None
	
	# For DINOv3 models, try to load with weights directly from torch.hub.load
	if DINO_MODEL.startswith("dinov3") and DINO_CKPT.exists():
		try:
			if DINO_MODEL == "dinov3_l":
				MODEL_NAME = "dinov3_vitl16"
			elif DINO_MODEL == "dinov3_b":
				MODEL_NAME = "dinov3_vitb16"
			elif DINO_MODEL == "dinov3_h":
				MODEL_NAME = "dinov3_vith16plus"
			else:
				raise ValueError(f"Unsupported DINO model: {DINO_MODEL}")
			
			torch.hub.set_dir('./checkpoints/dinov3')
			model = torch.hub.load(
				repo_or_dir='./dinov3',
				model=MODEL_NAME,
				source='local',
				weights=str(DINO_CKPT)
			)
			print(f"Loaded {DINO_MODEL} with weights from {DINO_CKPT}")
		except Exception as e:
			print(f"Failed to load {DINO_MODEL} with weights: {e}. Trying without weights.")
			model = None
	
	# Fallback: load from local repo without weights
	if model is None and DINO_REPO_DIR.exists():
		try:
			if DINO_MODEL == "dinov2_l":
				model = torch.hub.load(str(DINO_REPO_DIR), "dinov2_vitl14", source="local", pretrained=False)
			elif DINO_MODEL == "dinov3_l":
				model = torch.hub.load(str(DINO_REPO_DIR), "dinov3_vitl16", source="local", pretrained=False)
			elif DINO_MODEL == "dinov3_b":
				model = torch.hub.load(str(DINO_REPO_DIR), "dinov3_vitb16", source="local", pretrained=False)
			elif DINO_MODEL == "dinov3_h":
				model = torch.hub.load(str(DINO_REPO_DIR), "dinov3_vith16plus", source="local", pretrained=False)
		except Exception as e:
			print(f"Local DINO hub load failed: {e}. Falling back to remote hub.")
	
	# Fallback: load from remote hub
	if model is None:
		if DINO_MODEL == "dinov2_l":
			model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False)
		elif DINO_MODEL == "dinov3_l":
			model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", pretrained=False)
		elif DINO_MODEL == "dinov3_b":
			model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", pretrained=False)
		elif DINO_MODEL == "dinov3_h":
			model = torch.hub.load("facebookresearch/dinov3", "dinov3_vith16plus", pretrained=False)
		else:
			raise ValueError(f"Unsupported DINO model: {DINO_MODEL}")
	
	# Load checkpoint manually if not already loaded via torch.hub.load
	if DINO_CKPT.exists() and not DINO_MODEL.startswith("dinov3"):
		try:
			_load_state_dict_flex(model, DINO_CKPT)
		except Exception as e:
			print(f"Warning: failed to load checkpoint {DINO_CKPT}: {e}")
	elif not DINO_CKPT.exists():
		print(f"Warning: checkpoint not found at {DINO_CKPT}; using model weights as-in.")
	
	model.eval().to(device)
	return model


def build_transform(image_size: int = 518):
	# Minimal transforms to match ImageNet normalization expected by ViTs
	import torchvision.transforms as T

	return T.Compose(
		[
			T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
			T.CenterCrop(image_size),
			T.ToTensor(),
			T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		]
	)


@torch.no_grad()
def extract_patch_tokens(model, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
	"""
	Returns:
		patch_tokens: [N, C] tensor (N = H_patches*W_patches)
		grid_h, grid_w: patch grid dimensions
	"""
	if img_tensor.ndim == 3:
		img_tensor = img_tensor.unsqueeze(0)
	out = model.forward_features(img_tensor)
	if isinstance(out, dict) and "x_norm_patchtokens" in out:
		patch_tokens = out["x_norm_patchtokens"]  # [B, N, C]
	elif hasattr(model, "get_intermediate_layers"):
		patch_tokens = model.get_intermediate_layers(
			img_tensor, n=1, return_class_token=False
		)[0]  # [B, N, C]
	else:
		raise RuntimeError("Unable to obtain patch tokens from DINO model.")
	patch_tokens = patch_tokens[0]  # [N, C]
	num_tokens = patch_tokens.shape[0]
	grid_size = int(math.sqrt(num_tokens))
	return patch_tokens, grid_size, grid_size


def downsample_mask_to_patch_grid(mask: np.ndarray, gh: int, gw: int) -> torch.Tensor:
	"""
	mask: HxW binary uint8/0-1 np array
	return: [gh, gw] boolean tensor
	"""
	t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
	t = F.interpolate(t, size=(gh, gw), mode="nearest")
	return (t[0, 0] > 0.5)


def pool_tokens_by_mask(patch_tokens: torch.Tensor, mask_grid: torch.Tensor) -> Optional[torch.Tensor]:
	"""
	patch_tokens: [N, C]
	mask_grid: [gh, gw] bool
	"""
	gh, gw = mask_grid.shape
	mask_flat = mask_grid.flatten()
	if mask_flat.sum().item() == 0:
		return None
	selected = patch_tokens[mask_flat]  # [K, C]
	return selected.mean(dim=0)  # [C]


def iter_coco_instances(coco: COCO) -> Iterable[Tuple[dict, list]]:
	img_ids = coco.getImgIds()
	for img_id in img_ids:
		img_info = coco.loadImgs([img_id])[0]
		ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
		anns = coco.loadAnns(ann_ids)
		yield img_info, anns


def build_dataset_chunked(
	chunk_size: int = 5000,
	max_images: Optional[int] = None,
	min_area: float = 1.0,
) -> None:
	ensure_out_dir()
	ann_file = find_val_annotations_json()
	coco = COCO(str(ann_file))
	device = get_device()
	model = load_dinov2_model(device)
	transform = build_transform(518)

	cat_id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}
	with open(OUT_DIR / "category_id_to_name.json", "w") as f:
		json.dump(cat_id_to_name, f)

	features: List[np.ndarray] = []
	labels: List[int] = []
	chunk_index = 0
	processed_images = 0

	for img_info, anns in iter_coco_instances(coco):
		if max_images is not None and processed_images >= max_images:
			break
		processed_images += 1
		img_path = VAL_IMG_DIR / img_info["file_name"]
		if not img_path.exists():
			continue
		try:
			image = Image.open(img_path).convert("RGB")
		except Exception:
			continue

		img_tensor = transform(image).to(device)
		patch_tokens, gh, gw = extract_patch_tokens(model, img_tensor)

		for ann in anns:
			if ann.get("iscrowd", 0) == 1:
				continue
			if ann.get("area", 0.0) < min_area:
				continue
			mask = coco.annToMask(ann)  # HxW
			mask_grid = downsample_mask_to_patch_grid(mask, gh, gw)
			pooled = pool_tokens_by_mask(patch_tokens, mask_grid.to(patch_tokens.device))
			if pooled is None:
				continue
			features.append(pooled.detach().cpu().numpy().astype(np.float32))
			labels.append(int(ann["category_id"]))

			if len(features) >= chunk_size:
				save_chunk(features, labels, chunk_index)
				chunk_index += 1
				features, labels = [], []

		if processed_images % 100 == 0:
			print(f"Processed {processed_images} images...")

	if features:
		save_chunk(features, labels, chunk_index)


def save_chunk(features: List[np.ndarray], labels: List[int], chunk_index: int) -> None:
	ensure_out_dir()
	feat = np.stack(features, axis=0).astype(np.float16)
	labs = np.asarray(labels, dtype=np.int32)
	out_path = OUT_DIR / f"part_{chunk_index:05d}.npz"
	np.savez_compressed(out_path, features=feat, labels=labs)
	print(f"Saved {feat.shape[0]} samples -> {out_path}")


def load_all_chunks(limit_points: Optional[int] = None, class_filter: Optional[set] = None) -> Tuple[np.ndarray, np.ndarray]:
	files = sorted(OUT_DIR.glob("part_*.npz"))
	if not files:
		raise FileNotFoundError("No dataset chunks found. Run extraction first.")
	all_features: List[np.ndarray] = []
	all_labels: List[np.ndarray] = []
	for fp in files:
		with np.load(fp) as data:
			feat = data["features"]
			lab = data["labels"]
			if class_filter:
				mask = np.isin(lab, list(class_filter))
				if not mask.any():
					continue
				feat = feat[mask]
				lab = lab[mask]
			all_features.append(feat)
			all_labels.append(lab)
	features = np.concatenate(all_features, axis=0).astype(np.float32)
	labels = np.concatenate(all_labels, axis=0).astype(np.int32)
	if limit_points is not None and features.shape[0] > limit_points:
		rng = np.random.RandomState(0)
		idx = rng.choice(features.shape[0], size=limit_points, replace=False)
		features = features[idx]
		labels = labels[idx]
	return features, labels


def load_category_id_to_name() -> dict:
	fp = OUT_DIR / "category_id_to_name.json"
	if fp.exists():
		with open(fp, "r") as f:
			d = json.load(f)
		# keys may be strings; convert to int
		return {int(k): v for k, v in d.items()}
	# Fallback to annotations if json missing
	coco = COCO(str(find_val_annotations_json()))
	return {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}


def resolve_classes_arg(raw_list: Optional[List[str]], id_to_name: dict) -> Tuple[Optional[set], Optional[List[str]]]:
	"""
	raw_list may be like ["person,dog", "car"] or ["1", "17"]
	return (set_of_ids or None, ordered_names or None)
	"""
	if not raw_list:
		return None, None
	# flatten and split commas
	tokens: List[str] = []
	for item in raw_list:
		if item is None:
			continue
		for part in str(item).split(","):
			part = part.strip()
			if part:
				tokens.append(part)
	if not tokens:
		return None, None
	name_to_id = {v.lower(): k for k, v in id_to_name.items()}
	selected_ids: List[int] = []
	selected_names: List[str] = []
	for t in tokens:
		# Support CLI underscores for multi-word classes, e.g., "potted_plant"
		t = t.replace("_", " ")
		if t.isdigit():
			i = int(t)
			if i in id_to_name:
				selected_ids.append(i)
				selected_names.append(id_to_name[i])
			else:
				print(f"Warning: ID {i} not found in categories; skipping.")
		else:
			key = t.lower()
			if key in name_to_id:
				i = name_to_id[key]
				selected_ids.append(i)
				selected_names.append(id_to_name[i])
			else:
				print(f"Warning: class '{t}' not found; skipping.")
	if not selected_ids:
		return None, None
	return set(selected_ids), selected_names


def _slugify(name: str) -> str:
	# Prefer underscores for spaces to match CLI style
	name = name.replace(" ", "_")
	return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.lower()).strip("_-")

def run_tsne_and_plot(
	max_points: int = 50000,
	pca_dims: int = 50,
	perplexity: float = 30.0,
	random_state: int = 42,
	class_filter: Optional[set] = None,
	id_to_name: Optional[dict] = None,
	class_point_size: float = 60.0,
) -> Path:
	X, y = load_all_chunks(limit_points=max_points, class_filter=class_filter)
	print(f"Loaded dataset for TSNE: X={X.shape}, y={y.shape}")
	if PCA is not None and pca_dims > 0 and X.shape[1] > pca_dims:
		pca = PCA(n_components=pca_dims, random_state=random_state)
		X = pca.fit_transform(X)
		print(f"PCA -> {X.shape}")

	embedding = TSNE(
		n_components=2,
		perplexity=perplexity,
		initialization="pca",
		n_jobs=os.cpu_count() or 1,
		random_state=random_state,
		negative_gradient_method="bh",
	).fit(X)

	plt.figure(figsize=(9, 8))
	ax = plt.gca()
	ax.set_xticks([])
	ax.set_yticks([])
	if class_filter and id_to_name is not None:
		# Distinct, categorical colors with legend
		uniq = sorted(list(set(int(i) for i in np.unique(y))))
		cmap = plt.get_cmap("tab10")
		handles = []
		for idx, cid in enumerate(uniq):
			color = cmap(idx % 10)
			mask = y == cid
			pts = plt.scatter(
				embedding[mask, 0],
				embedding[mask, 1],
				s=class_point_size,
				color=color,
				alpha=0.6,
				label=id_to_name.get(int(cid), str(int(cid))),
			)
			handles.append(pts)
		plt.legend(
			handles=handles,
			loc="best",
			title="Class",
			markerscale=2,
			frameon=True,
			fontsize=14,
			title_fontsize=16,
		)
	else:
		# Default continuous colorbar for all classes
		sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=y, s=1.5, cmap="rainbow", alpha=0.8)
		cbar = plt.colorbar(sc)
		cbar.set_label("Label id")
	plt.tight_layout()
	# Build output name
	if class_filter and id_to_name:
		names = [_slugify(id_to_name[int(i)]) for i in sorted(list(class_filter))]
		if len(names) > 6:
			tag = "_".join(names[:6]) + f"_and_{len(names)-6}_more"
		else:
			tag = "_".join(names)
		filename = f"tsne_val2017_{tag}.png"
	else:
		filename = "tsne_val2017_all.png"
	out_path = PLOTS_DIR / filename
	plt.savefig(out_path, dpi=200)
	plt.close()
	print(f"Saved TSNE plot -> {out_path}")
	return out_path


def dataset_exists() -> bool:
	return any(OUT_DIR.glob("part_*.npz"))


def main():
	parser = argparse.ArgumentParser(description="COCO val2017 DINO TSNE by class")
	parser.add_argument("--extract", action="store_true", help="Force (re)extract features")
	parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images to process")
	parser.add_argument("--chunk-size", type=int, default=5000, help="Number of instances per saved shard")
	parser.add_argument("--max-points", type=int, default=50000, help="Max points to plot in TSNE for speed")
	parser.add_argument("--perplexity", type=float, default=30.0, help="TSNE perplexity")
	parser.add_argument("--pca-dims", type=int, default=50, help="PCA dims before TSNE (0 to disable)")
	parser.add_argument(
		"--classes",
		nargs="*",
		default=None,
		help="Optional list of COCO class names or IDs. Comma-separated or space-separated.",
	)
	parser.add_argument(
		"--class-point-size",
		type=float,
		default=60.0,
		help="Marker size (matplotlib 's') for per-class scatter when --classes is provided.",
	)
	args = parser.parse_args()

	ensure_out_dir()

	if not dataset_exists():
		if args.extract:
			print("Starting feature extraction...")
			build_dataset_chunked(chunk_size=args.chunk_size, max_images=args.max_images)
			print("Feature extraction complete.")
		else:
			raise FileNotFoundError(
				f"No precomputed features found in {OUT_DIR}. Run with --extract to generate them first."
			)

	pca_dims = args.pca_dims if args.pca_dims > 0 else 0
	id_to_name = load_category_id_to_name()
	class_filter, ordered_names = resolve_classes_arg(args.classes, id_to_name)
	out = run_tsne_and_plot(
		max_points=args.max_points,
		pca_dims=pca_dims,
		perplexity=args.perplexity,
		class_filter=class_filter,
		id_to_name=id_to_name,
		class_point_size=args.class_point_size,
	)


if __name__ == "__main__":
	main()


