import argparse
import os
import sys

from no_time_to_train.dataset.metainfo import METAINFO
COCO_CLASSES = METAINFO["few_shot_classes"]

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception as import_error:
    print(
        "Error: pycocotools is required to run this script. "
        "Install with: pip install pycocotools",
        file=sys.stderr,
    )
    raise


def evaluate_and_print_summary(
    coco_gt: COCO, predictions_path: str, iou_type: str, cat_ids: list[int]
) -> None:
    """
    Load predictions and run COCOeval for the given iou_type ('bbox' or 'segm'),
    then print the standard summary to stdout.
    """
    print("=" * 80)
    print(f"Evaluation type: {iou_type}")
    print(f"Predictions file: {predictions_path}")
    # Allow dataset-like JSON {"images":..., "annotations":[...], "categories":...} or plain list
    try:
        import json
        with open(predictions_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "annotations" in data and isinstance(data["annotations"], list):
                res_list = data["annotations"]
            else:
                raise ValueError("Unsupported results dict format: expected key 'annotations' with a list.")
        elif isinstance(data, list):
            res_list = data
        else:
            raise ValueError("Unsupported results JSON format: expected list or dict with 'annotations'.")
        coco_dt = coco_gt.loadRes(res_list)
    except Exception as e:
        print(f"Failed to load predictions from '{predictions_path}': {e}", file=sys.stderr)
        return

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    # Restrict evaluation to selected categories and all images indexed in coco_gt
    try:
        coco_eval.params.catIds = cat_ids
        coco_eval.params.imgIds = list(coco_gt.imgs.keys())
    except Exception:
        pass
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate COCO-format predictions (bbox and/or segm) against COCO-format ground truth "
            "and print the standard summaries."
        )
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to COCO-format ground truth annotations JSON (e.g., instances_val2017.json).",
    )
    parser.add_argument(
        "--pred",
        default=None,
        help="Path to COCO-format predictions JSON containing both bbox and segm results.",
    )
    # Backward compatibility: allow old flags; if provided, will fall back to them
    parser.add_argument("--pred-bbox", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pred-segm", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.gt):
        print(f"Ground truth file not found: {args.gt}", file=sys.stderr)
        sys.exit(1)

    predictions_path = args.pred or args.pred_bbox or args.pred_segm
    if not predictions_path:
        print(
            "Please provide --pred with a COCO-format predictions file containing bbox and segm.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=" * 80)
    print(f"Ground truth file: {args.gt}")
    # Load full GT, then filter to COCO_CLASSES (few-shot classes) like run_qwen_coco_detections.py
    coco_full = COCO(args.gt)
    cat_ids = coco_full.getCatIds(catNms=COCO_CLASSES)
    ann_ids = coco_full.getAnnIds(catIds=cat_ids)
    filtered_anns = coco_full.loadAnns(ann_ids)
    coco_gt = COCO()
    coco_gt.dataset = coco_full.dataset.copy()
    coco_gt.dataset["annotations"] = filtered_anns
    coco_gt.createIndex()

    if not os.path.isfile(predictions_path):
        print(f"Predictions file not found: {predictions_path}", file=sys.stderr)
        sys.exit(1)

    # Evaluate both bbox and segm from the same file if available
    evaluate_and_print_summary(coco_gt, predictions_path, "bbox", cat_ids)
    evaluate_and_print_summary(coco_gt, predictions_path, "segm", cat_ids)

    print("=" * 80)


if __name__ == "__main__":
    main()

