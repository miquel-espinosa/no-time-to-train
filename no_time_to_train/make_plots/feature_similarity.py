#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

# Optional matplotlib for colormaps only (no axes/figures saved)
try:
	import matplotlib
	matplotlib.use("Agg")
	from matplotlib import cm as mpl_cm  # type: ignore
except Exception:
	mpl_cm = None  # type: ignore

# --------------------------------------------------------------------------------------
# Paths and basic utils (mirrors tsne-coco.py for consistency)
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(".")
COCO_ROOT = Path("./data/coco")
ANN_ROOT = COCO_ROOT / "annotations"
VAL_IMG_DIR = COCO_ROOT / "val2017"
OUT_DIR = Path("no_time_to_train/make_plots/feature_similarity")
DINOV2_REPO_DIR = Path("./dinov2")
DINOV2_CKPT = Path("./checkpoints/dinov2/dinov2_vitl14_pretrain.pth")


def ensure_out_dirs() -> Dict[str, Path]:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	p = {
		"base": OUT_DIR,
		"heatmaps": OUT_DIR / "heatmaps",
		"heatmaps_prototype": OUT_DIR / "heatmaps_prototype",
	}
	for v in p.values():
		v.mkdir(parents=True, exist_ok=True)
	return p


def find_val_annotations_json() -> Path:
	if ANN_ROOT.exists():
		candidates = list(ANN_ROOT.glob("instances_val*2017.json"))
		candidates += list(ANN_ROOT.glob("instances_vale*2017.json"))
		if candidates:
			for p in candidates:
				if "instances_val2017" in p.stem:
					return p
			return candidates[0]
	raise FileNotFoundError(f"COCO annotations directory not found at {ANN_ROOT}")


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
	new_state = {}
	for k, v in state.items():
		if k.startswith("module."):
			k = k[len("module.") :]
		new_state[k] = v
	missing, unexpected = model.load_state_dict(new_state, strict=False)
	print(f"Loaded weights from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})")


def load_dinov2_model(device: torch.device):
	"""
	Load DINOv2 ViT-L/14 from local repo/checkpoint if available; fallback to hub.
	"""
	model = None
	if DINOV2_REPO_DIR.exists():
		try:
			model = torch.hub.load(str(DINOV2_REPO_DIR), "dinov2_vitl14", source="local", pretrained=False)
		except Exception as e:
			print(f"Local DINOv2 hub load failed: {e}. Falling back to remote hub.")
	if model is None:
		model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False)
	if DINOV2_CKPT.exists():
		try:
			_load_state_dict_flex(model, DINOV2_CKPT)
		except Exception as e:
			print(f"Warning: failed to load checkpoint {DINOV2_CKPT}: {e}")
	else:
		print(f"Warning: checkpoint not found at {DINOV2_CKPT}; using model weights as-in.")
	model.eval().to(device)
	return model


def build_transform(image_size: int = 518):
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
	if img_tensor.ndim == 3:
		img_tensor = img_tensor.unsqueeze(0)
	out = model.forward_features(img_tensor)
	if isinstance(out, dict) and "x_norm_patchtokens" in out:
		patch_tokens = out["x_norm_patchtokens"]  # [B, N, C]
	elif hasattr(model, "get_intermediate_layers"):
		patch_tokens = model.get_intermediate_layers(img_tensor, n=1, return_class_token=False)[0]
	else:
		raise RuntimeError("Unable to obtain patch tokens from DINOv2 model.")
	patch_tokens = patch_tokens[0]  # [N, C]
	num_tokens = patch_tokens.shape[0]
	grid_size = int(math.sqrt(num_tokens))
	return patch_tokens, grid_size, grid_size


def downsample_mask_to_patch_grid(mask: np.ndarray, gh: int, gw: int) -> torch.Tensor:
	t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
	t = F.interpolate(t, size=(gh, gw), mode="nearest")
	return (t[0, 0] > 0.5)


def iter_coco_instances(coco: COCO) -> Iterable[Tuple[dict, List[dict]]]:
	img_ids = coco.getImgIds()
	for img_id in img_ids:
		img_info = coco.loadImgs([img_id])[0]
		ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
		anns = coco.loadAnns(ann_ids)
		yield img_info, anns


# --------------------------------------------------------------------------------------
# Resizing helper to mirror DINOv2 preprocessing (Resize(shorter=image_size) + CenterCrop)
# --------------------------------------------------------------------------------------
def resize_for_dinov2_pil(image: Image.Image, image_size: int) -> Image.Image:
	import torchvision.transforms as T
	t = T.Compose(
		[
			T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
			T.CenterCrop(image_size),
		]
	)
	return t(image)


# --------------------------------------------------------------------------------------
# Mask utilities: resize mask like image, compute contour, and overlay on images
# --------------------------------------------------------------------------------------
def resize_mask_for_dinov2(mask_np: np.ndarray, image_size: int) -> np.ndarray:
	"""
	Resize + center-crop a binary mask to match DINOv2 preprocessing of images.
	returns: [H, W] bool numpy array with H=W=image_size
	"""
	from torchvision.transforms import InterpolationMode
	import torchvision.transforms.functional as TF

	mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255), mode="L")
	# Match image transform: Resize(shorter=image_size) + CenterCrop(image_size)
	# Using nearest interpolation for masks
	# First compute resize while preserving aspect with shorter side = image_size
	w, h = mask_img.size
	if w <= 0 or h <= 0:
		return np.zeros((image_size, image_size), dtype=bool)
	if w < h:
		new_w = image_size
		new_h = int(round(h * (image_size / w)))
	else:
		new_h = image_size
		new_w = int(round(w * (image_size / h)))
	mask_resized = mask_img.resize((new_w, new_h), resample=Image.NEAREST)
	# Center crop to square image_size
	left = max(0, (new_w - image_size) // 2)
	top = max(0, (new_h - image_size) // 2)
	right = left + image_size
	bottom = top + image_size
	mask_cropped = mask_resized.crop((left, top, right, bottom))
	out = np.array(mask_cropped)
	return (out > 127)


def compute_binary_outline(binary_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
	"""
	Compute a contour outline from a binary mask using simple morphological ops.
	returns: bool array where True indicates contour pixels.
	"""
	if binary_mask.size == 0:
		return binary_mask
	b = binary_mask.astype(bool)
	# 4-neighbor erosion
	eroded = (
		b
		& np.roll(b, 1, axis=0)
		& np.roll(b, -1, axis=0)
		& np.roll(b, 1, axis=1)
		& np.roll(b, -1, axis=1)
	)
	outline = b & (~eroded)
	# Thicken by simple dilation if requested
	for _ in range(max(0, thickness - 1)):
		outline = (
			outline
			| np.roll(outline, 1, axis=0)
			| np.roll(outline, -1, axis=0)
			| np.roll(outline, 1, axis=1)
			| np.roll(outline, -1, axis=1)
		)
	return outline


def overlay_contour_on_image(base_img_np: np.ndarray, contour_mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
	"""
	Draw contour pixels (True in contour_mask) over base image with the given color.
	base_img_np: HxWx3 uint8
	contour_mask: HxW bool
	"""
	if base_img_np.ndim != 3 or base_img_np.shape[2] != 3:
		raise ValueError("base_img_np must be HxWx3 uint8")
	h, w = base_img_np.shape[:2]
	if contour_mask.shape[0] != h or contour_mask.shape[1] != w:
		raise ValueError("contour_mask shape must match base image size")
	out = base_img_np.copy()
	yy, xx = np.where(contour_mask)
	out[yy, xx, 0] = color[0]
	out[yy, xx, 1] = color[1]
	out[yy, xx, 2] = color[2]
	return out


# --------------------------------------------------------------------------------------
# Image selection
# --------------------------------------------------------------------------------------
def select_images_with_multiple_instances(
	coco: COCO,
	target_cat_names: List[str],
	num_images: int = 30,
	min_area: float = 10000.0,
	max_area: float = 300000.0,
	min_instances: int = 2,
	seed: int = 0,
	max_per_class: Optional[int] = None,
) -> List[Tuple[dict, List[dict]]]:
	"""
	Select up to `num_images` unique images, balanced across the requested classes.
	An image is a candidate for class c if it contains at least `min_instances` instances
	of class c whose areas are within [min_area, max_area].
	"""
	cat_ids = coco.getCatIds(catNms=target_cat_names)
	cat_set = set(cat_ids)
	if len(cat_ids) == 0:
		return []

	# Gather candidates per class
	img_to_info_anns: Dict[int, Tuple[dict, List[dict]]] = {}
	candidates_per_class: Dict[int, List[int]] = {cid: [] for cid in cat_ids}

	for img_info, anns in iter_coco_instances(coco):
		img_id = int(img_info["id"])
		img_to_info_anns[img_id] = (img_info, anns)
		# Count per-class instances under area constraints
		counts: Dict[int, int] = {cid: 0 for cid in cat_ids}
		for ann in anns:
			if ann.get("iscrowd", 0) == 1:
				continue
			cid = int(ann.get("category_id", -1))
			if cid not in cat_set:
				continue
			area = float(ann.get("area", 0.0))
			if not (float(min_area) <= area <= float(max_area)):
				continue
			counts[cid] += 1
		for cid, cnt in counts.items():
			if cnt >= int(min_instances):
				candidates_per_class[cid].append(img_id)

	# Shuffle with seed for reproducibility
	rng = random.Random(int(seed))
	for cid in candidates_per_class:
		rng.shuffle(candidates_per_class[cid])

	# Decide per-class quota
	if max_per_class is not None and max_per_class > 0:
		per_class_quota = int(max_per_class)
	else:
		per_class_quota = max(1, math.ceil(int(num_images) / max(1, len(cat_ids))))

	# Round-robin selection to promote class diversity
	selected_ids: List[int] = []
	selected_set = set()
	class_cursors: Dict[int, int] = {cid: 0 for cid in cat_ids}
	class_counts: Dict[int, int] = {cid: 0 for cid in cat_ids}
	ordered_cids = list(cat_ids)
	rng.shuffle(ordered_cids)

	while len(selected_ids) < int(num_images):
		progress = False
		for cid in ordered_cids:
			if class_counts[cid] >= per_class_quota:
				continue
			cur = class_cursors[cid]
			cand_list = candidates_per_class[cid]
			# Advance cursor to next unused candidate
			while cur < len(cand_list) and cand_list[cur] in selected_set:
				cur += 1
			class_cursors[cid] = cur
			if cur >= len(cand_list):
				continue
			img_id = cand_list[cur]
			selected_ids.append(img_id)
			selected_set.add(img_id)
			class_counts[cid] += 1
			class_cursors[cid] += 1
			progress = True
			if len(selected_ids) >= int(num_images):
				break
		if not progress:
			# No more candidates for quotas; relax quotas and fill from any remaining candidates
			all_remaining: List[int] = []
			for cid in ordered_cids:
				cur = class_cursors[cid]
				all_remaining.extend([iid for iid in candidates_per_class[cid][cur:] if iid not in selected_set])
			rng.shuffle(all_remaining)
			for iid in all_remaining:
				if len(selected_ids) >= int(num_images):
					break
				selected_ids.append(iid)
				selected_set.add(iid)
			break

	return [img_to_info_anns[iid] for iid in selected_ids if iid in img_to_info_anns]


# --------------------------------------------------------------------------------------
# PCA visualization (full image, not masked)
# --------------------------------------------------------------------------------------
def compute_pca_rgb_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
	"""
	tokens: [N, C] float tensor on any device
	return: [N, 3] uint8 tensor in [0,255], PCA-minmax across all dims
	"""
	device = tokens.device
	X = tokens.to(dtype=torch.float32)
	mean = X.mean(dim=0, keepdim=True)
	Xc = X - mean
	# Use torch.pca_lowrank for top-3 PCs
	U, S, V = torch.pca_lowrank(Xc, q=3, center=False)  # center already done
	W = Xc @ V[:, :3]  # [N, 3]
	w_max = W.max()
	w_min = W.min()
	if (w_max - w_min) > 1e-6:
		Wn = (W - w_min) / (w_max - w_min)
	else:
		Wn = torch.zeros_like(W)
	RGB = (Wn * 255.0).clamp(0, 255).to(dtype=torch.uint8, device=device)
	return RGB


def tokens_rgb_canvas(rgb_tokens: torch.Tensor, gh: int, gw: int, patch_size: int = 14) -> np.ndarray:
	"""
	rgb_tokens: [N, 3] uint8, gh*gw == N
	return: HxWx3 uint8 numpy image (H=W=gh*patch_size)
	"""
	assert rgb_tokens.ndim == 2 and rgb_tokens.shape[1] == 3
	assert gh * gw == rgb_tokens.shape[0]
	H = gh * patch_size
	W = gw * patch_size
	canvas = torch.zeros((gh, gw, patch_size, patch_size, 3), dtype=torch.uint8, device=rgb_tokens.device)
	rgb_grid = rgb_tokens.reshape(gh, gw, 3)
	# Broadcast each token color to the patch square
	canvas[:] = rgb_grid[:, :, None, None, :]
	canvas = canvas.permute(0, 2, 1, 3, 4).reshape(H, W, 3).cpu().numpy()
	return canvas


# --------------------------------------------------------------------------------------
# Cosine similarity heatmaps
# --------------------------------------------------------------------------------------
def cosine_similarity_map(anchor: torch.Tensor, tokens: torch.Tensor, gh: int, gw: int, patch_size: int = 14) -> np.ndarray:
	"""
	anchor: [C]
	tokens: [N, C]
	return: heatmap HxWx3 uint8 image (using a colormap or gray)
	"""
	a = F.normalize(anchor[None, :], p=2, dim=-1)  # [1, C]
	t = F.normalize(tokens, p=2, dim=-1)  # [N, C]
	scores = (t @ a.t()).squeeze(1)  # [N], in [-1, 1]
	# Normalize to [0,1]
	s_min = float(scores.min().item())
	s_max = float(scores.max().item())
	if (s_max - s_min) > 1e-6:
		sn = (scores - s_min) / (s_max - s_min)
	else:
		sn = torch.zeros_like(scores)
	sn_img = sn.reshape(gh, gw).repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)  # [H,W]
	sn_np = sn_img.cpu().numpy()
	# Map to color
	if mpl_cm is not None:
		cmap = mpl_cm.get_cmap("inferno")
		color = (cmap(sn_np)[..., :3] * 255.0).astype(np.uint8)
	else:
		# Fallback: grayscale
		color = (sn_np[..., None] * 255.0).astype(np.uint8).repeat(3, axis=2)
	return color


def draw_red_cross(img_np: np.ndarray, center_xy: Tuple[int, int], size: int = 6, thickness: int = 2) -> np.ndarray:
	"""
	Draw a red cross at pixel coordinates (x, y) on an RGB uint8 numpy image.
	"""
	img = Image.fromarray(img_np)
	draw = ImageDraw.Draw(img)
	x, y = center_xy
	# Draw a thicker white cross underneath for contrast
	bg_thickness = max(thickness + 4, thickness * 2)
	for t in range(-(bg_thickness // 2), bg_thickness - (bg_thickness // 2)):
		draw.line([(x - size, y + t), (x + size, y + t)], fill=(255, 255, 255))
		draw.line([(x + t, y - size), (x + t, y + size)], fill=(255, 255, 255))
	# Draw the main red cross on top
	for t in range(-(thickness // 2), thickness - (thickness // 2)):
		# Horizontal
		draw.line([(x - size, y + t), (x + size, y + t)], fill=(255, 0, 0))
		# Vertical
		draw.line([(x + t, y - size), (x + t, y + size)], fill=(255, 0, 0))
	return np.array(img)


def grid_coord_from_mask(mask_grid: torch.Tensor) -> Optional[Tuple[int, int]]:
	"""
	mask_grid: [gh, gw] bool
	Returns integer (x, y) indices in grid (column, row) that are INSIDE the mask,
	choosing the in-mask patch closest to the centroid.
	"""
	ys, xs = torch.where(mask_grid)
	if ys.numel() == 0:
		return None
	# Centroid in continuous coordinates (row=y, col=x)
	cy = ys.float().mean()
	cx = xs.float().mean()
	# Among in-mask indices, pick the one closest to the centroid to ensure it's inside
	coords = torch.stack([xs.float(), ys.float()], dim=1)  # [K, 2] as (x, y)
	center = torch.stack([cx, cy], dim=0)  # (2,)
	dist2 = ((coords - center) ** 2).sum(dim=1)  # [K]
	best_idx = int(torch.argmin(dist2).item())
	return int(xs[best_idx].item()), int(ys[best_idx].item())


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------
def process_images(
	selected: List[Tuple[dict, List[dict]]],
	coco: COCO,
	model,
	device: torch.device,
	transform,
	out_dirs: Dict[str, Path],
	image_size: int = 518,
	patch_size: int = 14,
	min_area: float = 5000.0,
) -> None:
	# Map class IDs to names for naming
	id_to_name = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

	# Precompute per-class image id lists from the selected set to diversify pairings
	# and a round-robin cursor per class. Shuffle for variation but keep deterministic
	# order within a single run.
	class_to_img_ids: Dict[int, List[int]] = {}
	selected_imgid_to_info: Dict[int, dict] = {}
	for img_info, anns in selected:
		img_id = int(img_info["id"])
		selected_imgid_to_info[img_id] = img_info
		present_cids = {int(a.get("category_id", -1)) for a in anns if int(a.get("iscrowd", 0)) == 0}
		for cid in present_cids:
			class_to_img_ids.setdefault(cid, []).append(img_id)
	# Shuffle candidate lists so consecutive calls pick different targets
	rng_local = random.Random(0)
	for cid in list(class_to_img_ids.keys()):
		rng_local.shuffle(class_to_img_ids[cid])
	class_to_cursor: Dict[int, int] = {cid: 0 for cid in class_to_img_ids}

	for img_info, anns in selected:
		img_id = int(img_info["id"])
		img_path = VAL_IMG_DIR / img_info["file_name"]
		if not img_path.exists():
			continue
		try:
			image = Image.open(img_path).convert("RGB")
		except Exception:
			continue

		# Prepare originals: keep both full-res and resized (matching DINOv2 preprocessing)
		orig_resized = resize_for_dinov2_pil(image, image_size)

		# Extract tokens
		img_tensor = transform(image).to(device)
		tokens, gh, gw = extract_patch_tokens(model, img_tensor)  # [N, C]

		# PCA visualization across full image
		rgb_tokens = compute_pca_rgb_from_tokens(tokens)  # [N,3] u8
		pca_canvas = tokens_rgb_canvas(rgb_tokens, gh, gw, patch_size=patch_size)  # HxWx3

		# For heatmaps: choose an anchor inside one instance of a target class present in this image
		# Strategy: prefer classes with multiple instances and area >= min_area
		ann_candidates = [a for a in anns if a.get("iscrowd", 0) == 0 and float(a.get("area", 0.0)) >= float(min_area)]
		if not ann_candidates:
			continue

		# Prefer 'dog','cat','car','person' if available
		priority = ["dog", "cat", "car", "person"]
		name_to_id = {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}
		priority_ids = [name_to_id[n] for n in priority if n in name_to_id]

		priority_anns = [a for a in ann_candidates if int(a["category_id"]) in priority_ids]
		anchor_ann = priority_anns[0] if priority_anns else ann_candidates[0]
		anchor_cat_id = int(anchor_ann["category_id"])
		anchor_cat_name = id_to_name.get(anchor_cat_id, str(anchor_cat_id))
		anchor_ann_id = int(anchor_ann["id"])

		# Build patch-grid mask for anchor ann
		mask = coco.annToMask(anchor_ann)  # HxW (orig)
		# IMPORTANT: resize mask with same preprocessing as image before mapping to patch grid
		mask_resized_bool = resize_mask_for_dinov2(mask, image_size=image_size)  # [image_size, image_size] bool
		mask_grid = downsample_mask_to_patch_grid(mask_resized_bool.astype(np.uint8), gh, gw)  # [gh, gw] bool
		coord = grid_coord_from_mask(mask_grid)
		if coord is None:
			continue
		ax, ay = coord  # grid coordinates (x=col, y=row)
		anchor_idx = ay * gw + ax
		anchor_vec = tokens[anchor_idx]  # [C]

		# Intra-image similarity heatmap (mark red cross at anchor)
		heat_intra = cosine_similarity_map(anchor_vec, tokens, gh, gw, patch_size=patch_size)
		# Compute pixel coordinates of anchor
		ax_px = ax * patch_size + patch_size // 2
		ay_px = ay * patch_size + patch_size // 2
		heat_intra_marked = draw_red_cross(heat_intra, (ax_px, ay_px), size=18, thickness=6)

		# Compute class prototype by averaging all tokens within the selected mask
		mask_flat = mask_grid.reshape(-1)  # [N] bool
		if bool(mask_flat.sum().item() > 0):
			prototype_vec = tokens[mask_flat].mean(dim=0)  # [C]
		else:
			prototype_vec = anchor_vec

		# Find another image with the same category to compute inter-image similarity
		other_img_info = None
		candidates = class_to_img_ids.get(anchor_cat_id, [])
		if len(candidates) > 0:
			cur = class_to_cursor.get(anchor_cat_id, 0)
			# Try up to len(candidates) times to find an id different from current
			for _ in range(len(candidates)):
				oid = candidates[cur]
				cur = (cur + 1) % len(candidates)
				if int(oid) != img_id:
					other_img_info = selected_imgid_to_info.get(int(oid))
					break
			class_to_cursor[anchor_cat_id] = cur
		# If none chosen from selected (e.g., only one image for that class), fallback globally
		if other_img_info is None:
			global_ids = [int(oid) for oid in coco.getImgIds(catIds=[anchor_cat_id]) if int(oid) != img_id]
			if len(global_ids) > 0:
				oid = global_ids[rng_local.randrange(len(global_ids))]
				other_img_info = coco.loadImgs([int(oid)])[0]

		heat_inter = None
		other_img_id: Optional[int] = None
		other_tokens = None
		ogh = 0
		ogw = 0
		other_img = None
		other_resized = None
		if other_img_info is not None:
			other_img_id = int(other_img_info["id"])
			other_path = VAL_IMG_DIR / other_img_info["file_name"]
			if other_path.exists():
				try:
					other_img = Image.open(other_path).convert("RGB")
					other_resized = resize_for_dinov2_pil(other_img, image_size)
					other_tensor = transform(other_img).to(device)
					other_tokens, ogh, ogw = extract_patch_tokens(model, other_tensor)
					heat_inter = cosine_similarity_map(anchor_vec, other_tokens, ogh, ogw, patch_size=patch_size)
				except Exception:
					heat_inter = None

		# Save heatmaps under a dedicated folder to link the pair
		experiment_dir = out_dirs["heatmaps"] / f"{anchor_cat_name}_{img_id}_{anchor_ann_id}"
		experiment_dir.mkdir(parents=True, exist_ok=True)
		# Save originals and PCA in the same folder for easier navigation
		image.save(experiment_dir / f"original_full_{img_id}.png")
		orig_resized.save(experiment_dir / f"original_resized_{img_id}.png")
		Image.fromarray(pca_canvas).save(experiment_dir / f"pca_{img_id}.png")
		Image.fromarray(heat_intra_marked).save(experiment_dir / "sim_intra.png")
		if heat_inter is not None and other_img_id is not None:
			# Save target (other image) originals as well
			try:
				other_img.save(experiment_dir / f"target_full_{other_img_id}.png")
				other_resized.save(experiment_dir / f"target_resized_{other_img_id}.png")
			except Exception:
				pass
			Image.fromarray(heat_inter).save(experiment_dir / f"sim_inter_{other_img_id}.png")

		# Also save prototype-based heatmaps under a separate folder
		experiment_dir_proto = out_dirs["heatmaps_prototype"] / f"{anchor_cat_name}_{img_id}_{anchor_ann_id}"
		experiment_dir_proto.mkdir(parents=True, exist_ok=True)
		# Reuse originals and PCA for convenience
		image.save(experiment_dir_proto / f"original_full_{img_id}.png")
		orig_resized.save(experiment_dir_proto / f"original_resized_{img_id}.png")
		Image.fromarray(pca_canvas).save(experiment_dir_proto / f"pca_{img_id}.png")

		# Intra-image prototype similarity
		heat_intra_proto = cosine_similarity_map(prototype_vec, tokens, gh, gw, patch_size=patch_size)
		Image.fromarray(heat_intra_proto).save(experiment_dir_proto / "sim_intra.png")

		# Overlays: draw annotation mask contour on resized image and on PCA features
		try:
			# Resized image overlay
			mask_resized_bool = resize_mask_for_dinov2(mask, image_size=image_size)
			contour_resized = compute_binary_outline(mask_resized_bool, thickness=3)
			resized_np = np.array(orig_resized)
			resized_with_contour = overlay_contour_on_image(resized_np, contour_resized, color=(255, 0, 0))
			Image.fromarray(resized_with_contour).save(experiment_dir_proto / "mask_contour_on_resized.png")

			# PCA overlay: upsample grid mask to PCA canvas resolution
			mask_up = mask_grid.cpu().numpy().repeat(patch_size, axis=0).repeat(patch_size, axis=1)
			contour_pca = compute_binary_outline(mask_up, thickness=3)
			pca_with_contour = overlay_contour_on_image(pca_canvas, contour_pca, color=(255, 0, 0))
			Image.fromarray(pca_with_contour).save(experiment_dir_proto / "mask_contour_on_pca.png")
		except Exception:
			pass

		# Inter-image prototype similarity
		if other_tokens is not None and other_img_id is not None:
			heat_inter_proto = cosine_similarity_map(prototype_vec, other_tokens, ogh, ogw, patch_size=patch_size)
			# Save target (other image) originals as well
			try:
				if other_img is not None:
					other_img.save(experiment_dir_proto / f"target_full_{other_img_id}.png")
				if other_resized is not None:
					other_resized.save(experiment_dir_proto / f"target_resized_{other_img_id}.png")
			except Exception:
				pass
			Image.fromarray(heat_inter_proto).save(experiment_dir_proto / f"sim_inter_{other_img_id}.png")

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="DINOv2 feature similarity analysis on COCO images")
	parser.add_argument(
		"--classes",
		type=str,
		default="person,dog,cat,car",
		help="Comma-separated COCO class names to focus on.",
	)
	parser.add_argument("--num-images", type=int, default=30, help="Total number of images to select")
	parser.add_argument("--min-area", type=float, default=10000.0, help="Minimum annotation area (pixels)")
	parser.add_argument("--max-area", type=float, default=300000.0, help="Maximum annotation area (pixels)")
	parser.add_argument("--min-instances", type=int, default=2, help="Minimum instances per class in an image")
	parser.add_argument("--max-per-class", type=int, default=0, help="Optional cap per class (0 means auto-balance)")
	parser.add_argument("--seed", type=int, default=0, help="Random seed for selection shuffling")
	parser.add_argument("--image-size", type=int, default=518, help="Input size for DINOv2 (multiple of 14)")
	parser.add_argument("--patch-size", type=int, default=14, help="DINOv2 patch size (ViT-14)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	target_classes = [s.strip() for s in args.classes.split(",") if s.strip()]
	out_dirs = ensure_out_dirs()

	ann_file = find_val_annotations_json()
	coco = COCO(str(ann_file))
	device = get_device()
	model = load_dinov2_model(device)
	transform = build_transform(args.image_size)

	selected = select_images_with_multiple_instances(
		coco,
		target_classes,
		num_images=args.num_images,
		min_area=args.min_area,
		max_area=args.max_area,
		min_instances=args.min_instances,
		seed=args.seed,
		max_per_class=(args.max_per_class if args.max_per_class and args.max_per_class > 0 else None),
	)
	if not selected:
		print("No images found matching the selection criteria.")
		return
	print(f"Selected {len(selected)} images.")

	process_images(
		selected,
		coco,
		model,
		device,
		transform,
		out_dirs,
		image_size=args.image_size,
		patch_size=args.patch_size,
		min_area=args.min_area,
	)
	print(f"Saved outputs under: {OUT_DIR}")


if __name__ == "__main__":
	main()


