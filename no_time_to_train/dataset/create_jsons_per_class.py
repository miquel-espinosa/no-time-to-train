import argparse
import json
import os
import re
from typing import Dict, List, Any, Set
import random


def read_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_class_name(name: str) -> str:
    """
    Create a filesystem-friendly version of a COCO category name.
    Example: 'hair drier' -> 'hair_drier'
    """
    name = name.lower().strip()
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def filter_annotations_by_category(
    annotations: List[Dict[str, Any]], category_id: int
) -> List[Dict[str, Any]]:
    return [ann for ann in annotations if ann.get("category_id") == category_id]


def collect_image_ids_from_annotations(
    annotations: List[Dict[str, Any]]
) -> Set[int]:
    return {ann["image_id"] for ann in annotations}


def filter_images_by_ids(
    images: List[Dict[str, Any]], image_ids: Set[int]
) -> List[Dict[str, Any]]:
    keep_ids = image_ids
    return [img for img in images if img.get("id") in keep_ids]


def build_per_class_coco(
    info: Dict[str, Any],
    licenses: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    annotations: List[Dict[str, Any]],
    category: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a valid COCO-style dict for a single class.
    We include only the specified single category in 'categories'.
    """
    return {
        "info": info or {},
        "licenses": licenses or [],
        "images": images or [],
        "annotations": annotations or [],
        "categories": [category],
    }


def create_per_class_annotations(
    input_dir: str,
    output_dir: str,
    train_filename: str = "instances_train2017.json",
    val_filename: str = "instances_val2017.json",
    val_pos_count: int = 50,
    val_neg_count: int = 25,
    seed: int = 0,
) -> None:
    # Load source annotations
    train_path = os.path.join(input_dir, train_filename)
    val_path = os.path.join(input_dir, val_filename)

    train_data = read_json(train_path)
    val_data = read_json(val_path)

    # Prepare output directory
    ensure_dir(output_dir)

    # Use categories from train (COCO categories are consistent across splits)
    categories: List[Dict[str, Any]] = train_data.get("categories", [])
    if not categories:
        raise ValueError("No 'categories' found in training annotations.")

    # Common lookups
    train_info = train_data.get("info", {})
    train_licenses = train_data.get("licenses", [])
    train_images_all = train_data.get("images", [])
    train_annotations_all = train_data.get("annotations", [])

    val_info = val_data.get("info", {})
    val_licenses = val_data.get("licenses", [])
    val_images_all = val_data.get("images", [])
    val_annotations_all = val_data.get("annotations", [])

    for category in categories:
        category_id = category.get("id")
        category_name = category.get("name", f"category_{category_id}")
        safe_name = sanitize_class_name(category_name)

        # Create class-specific directory
        class_dir = os.path.join(output_dir, safe_name)
        ensure_dir(class_dir)

        # Train split: keep only images that contain this category, and annotations for this category
        train_annos = filter_annotations_by_category(train_annotations_all, category_id)
        train_image_ids = collect_image_ids_from_annotations(train_annos)
        train_images = filter_images_by_ids(train_images_all, train_image_ids)
        train_coco = build_per_class_coco(
            info=train_info,
            licenses=train_licenses,
            images=train_images,
            annotations=train_annos,
            category=category,
        )
        train_out_path = os.path.join(class_dir, "instances_train2017.json")
        with open(train_out_path, "w") as f:
            json.dump(train_coco, f, ensure_ascii=False, indent=2)

        # Val split: keep all images, but only annotations for this category
        val_annos = filter_annotations_by_category(val_annotations_all, category_id)
        val_coco = build_per_class_coco(
            info=val_info,
            licenses=val_licenses,
            images=val_images_all,
            annotations=val_annos,
            category=category,
        )
        val_out_path = os.path.join(class_dir, "instances_val2017.json")
        with open(val_out_path, "w") as f:
            json.dump(val_coco, f, ensure_ascii=False, indent=2)

        # Reduced Val split: select a subset of images for quicker evaluation
        # - Include up to 'val_pos_count' images that contain the category
        # - Include up to 'val_neg_count' images that do NOT contain the category
        # - Always include all annotations of this category for the selected images
        pos_image_ids = collect_image_ids_from_annotations(val_annos)
        all_val_image_ids = {img["id"] for img in val_images_all}
        neg_image_ids = all_val_image_ids - pos_image_ids

        # Use a per-category RNG for stability across categories
        rng = random.Random(seed + int(category_id or 0))
        # Convert sets to sorted lists for deterministic sampling across Python versions
        pos_pool = sorted(pos_image_ids)
        neg_pool = sorted(neg_image_ids)

        pos_k = min(val_pos_count, len(pos_pool))
        neg_k = min(val_neg_count, len(neg_pool))

        sampled_pos_ids = set(rng.sample(pos_pool, pos_k)) if pos_k > 0 else set()
        sampled_neg_ids = set(rng.sample(neg_pool, neg_k)) if neg_k > 0 else set()
        reduced_image_ids = sampled_pos_ids | sampled_neg_ids

        reduced_val_images = filter_images_by_ids(val_images_all, reduced_image_ids)
        # Only keep annotations for this category in the selected images
        reduced_val_annos = [
            ann for ann in val_annos if ann.get("image_id") in reduced_image_ids
        ]
        reduced_val_coco = build_per_class_coco(
            info=val_info,
            licenses=val_licenses,
            images=reduced_val_images,
            annotations=reduced_val_annos,
            category=category,
        )
        reduced_val_out_path = os.path.join(class_dir, "reduced_instances_val2017.json")
        with open(reduced_val_out_path, "w") as f:
            json.dump(reduced_val_coco, f, ensure_ascii=False, indent=2)

        print(
            f"[{category_name}] "
            f"train: {len(train_images)} images, {len(train_annos)} annotations -> {train_out_path}"
        )
        print(
            f"[{category_name}] "
            f"val:   {len(val_images_all)} images, {len(val_annos)} annotations -> {val_out_path}"
        )
        print(
            f"[{category_name}] "
            f"reduced val: {len(reduced_val_images)} images "
            f"(pos {len(sampled_pos_ids)}, neg {len(sampled_neg_ids)}), "
            f"{len(reduced_val_annos)} annotations -> {reduced_val_out_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create COCO-style per-class annotation JSONs.\n"
            "- Train: keep only images containing the class and its annotations.\n"
            "- Val: keep all images, only annotations for the class.\n"
            "- Reduced Val: sample a subset of images (pos/neg) with reproducible seed."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/coco/annotations",
        help="Directory containing COCO annotation files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/coco/annotations/per_class_instances",
        help="Directory to write per-class JSON files.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="instances_train2017.json",
        help="Training annotations filename.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="instances_val2017.json",
        help="Validation annotations filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible reduced val sampling.",
    )
    parser.add_argument(
        "--val_pos_count",
        type=int,
        default=50,
        help="Number of val images containing the category for reduced set.",
    )
    parser.add_argument(
        "--val_neg_count",
        type=int,
        default=25,
        help="Number of val images not containing the category for reduced set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_per_class_annotations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_filename=args.train_file,
        val_filename=args.val_file,
        val_pos_count=args.val_pos_count,
        val_neg_count=args.val_neg_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


