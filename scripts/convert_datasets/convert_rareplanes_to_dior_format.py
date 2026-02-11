#!/usr/bin/env python3
"""
Convert RarePlanes dataset to DIOR format.

RarePlanes format:
original_files/
├── train/
│   ├── PS-RGB_tiled/
│   │   ├── *.png
│   │   └── *.png.aux.xml
│   └── geojson_aircraft_tiled/
│       └── *.geojson
└── test/
    ├── PS-RGB_tiled/
    └── geojson_aircraft_tiled/

Annotations:
- GeoJSON polygons in WGS84 (lon/lat)
- Bounding boxes derived via GeoTransform from .aux.xml
- Category = properties["role"] (spaces replaced with dashes)

DIOR output:
/
├── info.txt
├── train/
├── test/
└── annotations/
    ├── train.json
    └── test.json
"""

"""
Usage: 

# roles are used as categories
python3 scripts/convert_datasets/convert_rareplanes_to_dior_format.py \
    --rareplanes_dir ./data/RAREPLANES

# all roles are mapped to a single 'airplane' class
python3 scripts/convert_datasets/convert_rareplanes_to_dior_format.py \
    --rareplanes_dir ./data/RAREPLANES \
    --output_dir ./data/RAREPLANES_SINGLE_CLASS \
    --single-class
"""

import os
import json
import argparse
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image
from tqdm import tqdm


def die(msg):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read_geotransform(aux_xml_path):
    tree = ET.parse(aux_xml_path)
    root = tree.getroot()
    gt_elem = root.find("GeoTransform")
    if gt_elem is None:
        die(f"GeoTransform not found in {aux_xml_path}")
    gt = [float(x) for x in gt_elem.text.split(",")]
    if len(gt) != 6:
        die(f"Invalid GeoTransform in {aux_xml_path}")
    return gt


def lonlat_to_pixel(lon, lat, gt):
    gt0, gt1, gt2, gt3, gt4, gt5 = gt
    if gt2 != 0 or gt4 != 0:
        die("Non-zero rotation terms in GeoTransform are not supported")
    x = (lon - gt0) / gt1
    y = (lat - gt3) / gt5
    return x, y


def polygon_to_bbox_pixels(coords, gt):
    xs, ys = [], []
    for lon, lat in coords:
        px, py = lonlat_to_pixel(lon, lat, gt)
        xs.append(px)
        ys.append(py)
    return min(xs), min(ys), max(xs), max(ys)


def normalize_role(role):
    return role.replace(" ", "-")


def collect_categories(splits, single_class=False):
    if single_class:
        categories = [{
            "id": 1,
            "name": "airplane",
            "supercategory": "aircraft"
        }]
        role_to_id = None  # not used in single-class mode
        return categories, role_to_id

    roles = set()
    for split in splits:
        geojson_dir = split / "geojson_aircraft_tiled"
        for gj in geojson_dir.glob("*.geojson"):
            with open(gj) as f:
                data = json.load(f)
            if not data.get("features"):
                die(f"Empty GeoJSON: {gj}")
            for feat in data["features"]:
                role = feat["properties"].get("role")
                if role is None:
                    die(f"Missing 'role' in {gj}")
                roles.add(normalize_role(role))
    roles = sorted(roles)
    categories = [
        {"id": i + 1, "name": r, "supercategory": "aircraft"}
        for i, r in enumerate(roles)
    ]
    role_to_id = {c["name"]: c["id"] for c in categories}
    return categories, role_to_id


def clip_polygon_to_tile(polygon, tile_x, tile_y, tile_size):
    """
    Clip a polygon to a tile and convert to tile-relative coordinates.
    
    Uses Sutherland-Hodgman algorithm for polygon clipping.
    
    Args:
        polygon: List of [x1, y1, x2, y2, ...] coordinates
        tile_x, tile_y: Top-left corner of tile in original image
        tile_size: Size of the tile
    
    Returns:
        Clipped polygon in tile coordinates or None if invalid
    """
    if len(polygon) < 6:  # Need at least 3 points
        return None
    
    # Convert flat list to list of (x, y) tuples
    points = [(polygon[i] - tile_x, polygon[i+1] - tile_y) 
              for i in range(0, len(polygon), 2)]
    
    # Clip against each edge of the tile
    def clip_edge(points, edge_func, inside_func):
        """Clip polygon against one edge."""
        if not points:
            return []
        
        output = []
        for i in range(len(points)):
            current = points[i]
            next_pt = points[(i + 1) % len(points)]
            
            current_inside = inside_func(current)
            next_inside = inside_func(next_pt)
            
            if current_inside:
                output.append(current)
                if not next_inside:
                    output.append(edge_func(current, next_pt))
            elif next_inside:
                output.append(edge_func(current, next_pt))
        
        return output
    
    def intersect_left(p1, p2):
        if p2[0] == p1[0]:
            return (0, p1[1])
        t = (0 - p1[0]) / (p2[0] - p1[0])
        return (0, p1[1] + t * (p2[1] - p1[1]))
    
    def intersect_right(p1, p2):
        if p2[0] == p1[0]:
            return (tile_size, p1[1])
        t = (tile_size - p1[0]) / (p2[0] - p1[0])
        return (tile_size, p1[1] + t * (p2[1] - p1[1]))
    
    def intersect_top(p1, p2):
        if p2[1] == p1[1]:
            return (p1[0], 0)
        t = (0 - p1[1]) / (p2[1] - p1[1])
        return (p1[0] + t * (p2[0] - p1[0]), 0)
    
    def intersect_bottom(p1, p2):
        if p2[1] == p1[1]:
            return (p1[0], tile_size)
        t = (tile_size - p1[1]) / (p2[1] - p1[1])
        return (p1[0] + t * (p2[0] - p1[0]), tile_size)
    
    # Clip against all four edges
    points = clip_edge(points, intersect_left, lambda p: p[0] >= 0)
    points = clip_edge(points, intersect_right, lambda p: p[0] <= tile_size)
    points = clip_edge(points, intersect_top, lambda p: p[1] >= 0)
    points = clip_edge(points, intersect_bottom, lambda p: p[1] <= tile_size)
    
    if len(points) < 3:
        return None
    
    # Convert back to flat list
    result = []
    for x, y in points:
        result.extend([float(x), float(y)])
    
    return result


def polygon_area(polygon):
    """Calculate area of polygon using shoelace formula."""
    if len(polygon) < 6:
        return 0
    
    n = len(polygon) // 2
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[2*i] * polygon[2*j + 1]
        area -= polygon[2*j] * polygon[2*i + 1]
    
    return abs(area) / 2


def polygon_bbox(polygon):
    """Get bounding box of polygon as [x, y, w, h]."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def clip_box_to_tile(bbox_xywh, tile_x, tile_y, tile_size):
    """
    Clip a bounding box to a tile and convert to tile-relative coordinates.
    
    Args:
        bbox_xywh: [x, y, width, height] in original image coordinates
        tile_x, tile_y: Top-left corner of tile in original image
        tile_size: Size of the tile
    
    Returns:
        Clipped bbox in tile coordinates [x, y, w, h] or None if invalid
    """
    x, y, w, h = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Convert to tile-relative coordinates
    x1_rel = x1 - tile_x
    y1_rel = y1 - tile_y
    x2_rel = x2 - tile_x
    y2_rel = y2 - tile_y
    
    # Clip to tile boundaries
    x1_clip = max(0, x1_rel)
    y1_clip = max(0, y1_rel)
    x2_clip = min(tile_size, x2_rel)
    y2_clip = min(tile_size, y2_rel)
    
    # Check if box is valid (has positive area)
    if x2_clip <= x1_clip or y2_clip <= y1_clip:
        return None
    
    # Calculate how much of the original box is preserved
    original_area = w * h
    clipped_w = x2_clip - x1_clip
    clipped_h = y2_clip - y1_clip
    clipped_area = clipped_w * clipped_h
    
    # Skip if less than 30% of original box is in tile
    if original_area > 0 and clipped_area / original_area < 0.3:
        return None
    
    # Skip very small boxes (less than 4x4 pixels)
    if clipped_w < 4 or clipped_h < 4:
        return None
    
    return [float(x1_clip), float(y1_clip), float(clipped_w), float(clipped_h)]


def process_split(
    split_name,
    split_dir,
    out_img_dir,
    role_to_id,
    start_image_id=1,
    start_ann_id=1,
):
    images = []
    annotations = []
    image_id = start_image_id
    ann_id = start_ann_id

    img_dir = split_dir / "PS-RGB_tiled"
    geojson_dir = split_dir / "geojson_aircraft_tiled"

    for img_path in tqdm(sorted(img_dir.glob("*.png")), desc=f"Processing {split_name}"):
        base = img_path.stem
        geojson_path = geojson_dir / f"{base}.geojson"
        aux_path = img_path.with_suffix(img_path.suffix + ".aux.xml")

        if not geojson_path.exists():
            die(f"Missing GeoJSON for image {img_path}")
        if not aux_path.exists():
            die(f"Missing aux.xml for image {img_path}")

        with Image.open(img_path) as img:
            width, height = img.size

        with open(geojson_path) as f:
            data = json.load(f)

        if not data.get("features"):
            die(f"No annotations in {geojson_path}")

        gt = read_geotransform(aux_path)

        dst_img_path = out_img_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        images.append({
            "id": image_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        for feat in data["features"]:
            geom = feat.get("geometry")
            if geom is None or geom.get("type") != "Polygon":
                die(f"Invalid geometry in {geojson_path}")

            coords = geom.get("coordinates")
            if not coords or not coords[0]:
                die(f"Empty polygon in {geojson_path}")

            # Convert polygon to flat pixel list
            flat_poly = []
            for lon, lat in coords[0]:
                px, py = lonlat_to_pixel(lon, lat, gt)
                flat_poly.extend([px, py])
            
            clipped_poly = clip_polygon_to_tile(
                flat_poly,
                tile_x=0,
                tile_y=0,
                tile_size=width
            )

            if clipped_poly is None:
                print(
                    f"WARNING: Polygon fully outside tile in {geojson_path} — skipping annotation",
                    file=sys.stderr
                )
                continue
            
            bbox = polygon_bbox(clipped_poly)
            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                print(
                    f"WARNING: Degenerate bbox after clipping in {geojson_path} — skipping annotation",
                    file=sys.stderr
                )
                continue
            
            if role_to_id is None:
                category_id = 1
            else:
                role = normalize_role(feat["properties"].get("role"))
                if role not in role_to_id:
                    die(f"Unknown role '{role}' in {geojson_path}")
                category_id = role_to_id[role]

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [],
                "iscrowd": 0,
                "ignore": 0,
            })
            ann_id += 1

        image_id += 1

    return images, annotations, image_id, ann_id


def main():
    parser = argparse.ArgumentParser(
        description="Convert RarePlanes dataset to DIOR format"
    )
    parser.add_argument(
        "--rareplanes_dir",
        type=str,
        required=True,
        help="Path to RarePlanes dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as rareplanes_dir)",
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Collapse all categories into a single 'airplane' class"
    )
    args = parser.parse_args()

    rareplanes_dir = Path(args.rareplanes_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = rareplanes_dir

    original_files = rareplanes_dir / "original_files"
    if not original_files.exists():
        die("original_files directory not found")

    train_src = original_files / "train"
    test_src = original_files / "test"

    for p in [train_src, test_src]:
        if not p.exists():
            die(f"Missing split directory: {p}")

    annotations_dir = output_dir / "annotations"
    train_out = output_dir / "train"
    test_out = output_dir / "test"

    annotations_dir.mkdir(parents=True, exist_ok=True)
    train_out.mkdir(exist_ok=True)
    test_out.mkdir(exist_ok=True)

    print("Collecting categories...")
    categories, role_to_id = collect_categories([train_src, test_src], single_class=args.single_class)
    print(f"Found {len(categories)} categories")

    print("\n=== Processing TRAIN split ===")
    train_images, train_anns, next_img_id, next_ann_id = process_split(
        "train",
        train_src,
        train_out,
        role_to_id,
        start_image_id=1,
        start_ann_id=1,
    )

    print("\n=== Processing TEST split ===")
    test_images, test_anns, _, _ = process_split(
        "test",
        test_src,
        test_out,
        role_to_id,
        start_image_id=1,
        start_ann_id=1,
    )

    train_json = {
        "type": "instances",
        "images": train_images,
        "annotations": train_anns,
        "categories": categories,
    }

    test_json = {
        "type": "instances",
        "images": test_images,
        "annotations": test_anns,
        "categories": categories,
    }

    with open(annotations_dir / "train.json", "w") as f:
        json.dump(train_json, f)

    with open(annotations_dir / "test.json", "w") as f:
        json.dump(test_json, f)

    # info.txt
    info_path = output_dir / "info.txt"
    with open(info_path, "w") as f:
        f.write(
f"""Dataset: RarePlanes
Task: Aircraft detection
Annotation type: Bounding boxes (converted from GeoJSON)
Label source: properties["role"]

Category mode: {'Single-class (all aircraft mapped to "airplane")' if args.single_class else 'Multi-class (each aircraft mapped to its own class)'}

Number of classes: {len(categories)}
Classes:
{chr(10).join("  - " + c["name"] for c in categories)}

Train split:
  Images: {len(train_images)}
  Annotations: {len(train_anns)}

Test split:
  Images: {len(test_images)}
  Annotations: {len(test_anns)}

Images:
  Format: PNG
  Resolution: Read per image (tiles expected to be uniform)

Notes:
- GeoJSON coordinates projected from WGS84 using GeoTransform in .aux.xml
- Bounding boxes only (no instance segmentation)
- Strict validation: no clipping, no silent skipping

Citation:
@misc{{RarePlanes_Dataset,
  title={{RarePlanes Dataset}},
  author={{Shermeyer, Jacob and Hossler, Thomas and Van Etten, Adam and Hogan, Daniel and Lewis, Ryan and Kim, Daeil}},
  organization={{In-Q-Tel - CosmiQ Works and AI.Reverie}},
  month={{June}},
  year={{2020}}
}}
"""
        )

    print("\n" + "=" * 60)
    print("RarePlanes conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - annotations/train.json ({len(train_images)} images, {len(train_anns)} annotations)")
    print(f"  - annotations/test.json ({len(test_images)} images, {len(test_anns)} annotations)")
    print(f"  - train/ ({len(train_images)} images)")
    print(f"  - test/ ({len(test_images)} images)")


if __name__ == "__main__":
    main()
