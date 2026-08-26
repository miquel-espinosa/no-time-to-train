#!/usr/bin/env python3
"""Prepare a small COCO val subset for local smoke tests.

Does not download the full train2017 / val2017 image zips. Uses existing
`instances_val2017.json` when present; otherwise downloads only the
annotations archive, samples N images, and fetches those JPEGs one by one.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ANN_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
VAL_IMAGE_URL = "http://images.cocodataset.org/val2017/{filename}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small COCO val subset under data/coco/"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/coco"),
        help="COCO root directory",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of val images to keep",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--keep-extra-images",
        action="store_true",
        help="Do not delete val images that are outside the subset",
    )
    return parser.parse_args()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def ensure_val_annotations(data_root: Path) -> Path:
    ann_path = data_root / "annotations" / "instances_val2017.json"
    if ann_path.exists():
        return ann_path

    zip_path = data_root / "annotations_trainval2017.zip"
    if not zip_path.exists():
        download(ANN_ZIP_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("annotations/instances_val2017.json", data_root)
    zip_path.unlink(missing_ok=True)
    if not ann_path.exists():
        raise FileNotFoundError(f"Failed to extract {ann_path}")
    return ann_path


def sample_coco(data: dict, size: int, seed: int) -> dict:
    images = list(data["images"])
    if len(images) <= size:
        return data

    rng = random.Random(seed)
    sampled = rng.sample(images, size)
    keep_ids = {img["id"] for img in sampled}
    annotations = [
        ann for ann in data.get("annotations", []) if ann["image_id"] in keep_ids
    ]
    return {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": sampled,
        "annotations": annotations,
    }


def download_images(images: list[dict], img_dir: Path) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images, 1):
        dest = img_dir / img["file_name"]
        if dest.exists():
            continue
        url = VAL_IMAGE_URL.format(filename=img["file_name"])
        print(f"[{i}/{len(images)}] {img['file_name']}")
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}", file=sys.stderr)


def prune_extra_images(img_dir: Path, keep_names: set[str]) -> int:
    if not img_dir.exists():
        return 0
    removed = 0
    for path in img_dir.iterdir():
        if path.is_file() and path.name not in keep_names:
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    img_dir = data_root / "val2017"
    ann_path = ensure_val_annotations(data_root)

    with ann_path.open() as f:
        coco = json.load(f)

    subset = sample_coco(coco, args.size, args.seed)
    download_images(subset["images"], img_dir)

    keep_names = {img["file_name"] for img in subset["images"]}
    if not args.keep_extra_images:
        removed = prune_extra_images(img_dir, keep_names)
        if removed:
            print(f"Removed {removed} extra images from {img_dir}")

    tmp_path = ann_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(subset, f)
    shutil.move(str(tmp_path), str(ann_path))

    n_img = len(subset["images"])
    n_ann = len(subset["annotations"])
    print(f"Wrote {ann_path} ({n_img} images, {n_ann} annotations)")
    print(f"Images: {img_dir}")


if __name__ == "__main__":
    main()
