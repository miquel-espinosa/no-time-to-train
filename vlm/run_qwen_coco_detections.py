#!/usr/bin/env python3
"""
run_qwen_coco_multi_gpu.py

One-process-per-GPU inference for Qwen2-VL -> COCO detections.

Usage:
    python run_qwen_coco_multi_gpu.py
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import List
import multiprocessing as mp
import argparse

from no_time_to_train.dataset.metainfo import METAINFO

COCO_CLASSES = METAINFO["few_shot_classes"]

# ----------------------------
# USER CONFIG
# ----------------------------
# Base folder where HF cached structure sits (your provided tree)
HF_CACHE_BASE = "./claptrap_checkpoints/qwen"

# COCO paths
COCO_ROOT = "claptrap_data/coco"
IMAGE_DIR = os.path.join(COCO_ROOT, "val2017")
ANNOTATIONS = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
VIS_DIR = "./vlm/qwen_outputs/qwen_visualizations"

# Output
OUTPUT_JSON = "./vlm/qwen_outputs/qwen_coco_results.json"
PARTIAL_RESULTS_DIR = "./vlm/qwen_outputs/qwen_partial_results"

# Inference config
BATCH_SIZE = 8                # per-GPU batch size
MAX_NEW_TOKENS = 256
MODEL_ID_OR_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # used only if snapshot not found (optional snapshot download)

# ----------------------------
# Utilities: find HF snapshot folder
# ----------------------------
def find_snapshot_dir(base_dir: str) -> str:
    """Find a directory under base_dir that contains model.safetensors index + config.json etc."""
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"{base_dir} does not exist.")
    for root, dirs, files in os.walk(base_dir):
        files_set = set(files)
        # check for indicators of a snapshot folder
        if "config.json" in files_set and (
            "model.safetensors.index.json" in files_set
            or any(f.startswith("pytorch_model") for f in files)
            or any(f.endswith(".safetensors") for f in files)
        ):
            return root
    raise FileNotFoundError(f"No valid model snapshot folder found under {base_dir}.")


def _visualize_batch(img, decoded_json_str, save_path):
    """
    Draw bounding boxes on a PIL image and save it.
    """
    from PIL import ImageDraw
    import json

    try:
        pred = json.loads(decoded_json_str)
    except:
        return

    objs = pred.get("objects", [])
    draw = ImageDraw.Draw(img)

    for o in objs:
        try:
            x1, y1, x2, y2 = o["box"]
        except:
            continue
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        label = o.get("label", "")
        conf = o.get("confidence", 0)
        draw.text((x1, y1), f"{label} {conf:.2f}", fill="red")

    img.save(save_path)



# ----------------------------
# Worker function (runs in child process)
# Important: import heavy libs inside worker AFTER setting CUDA_VISIBLE_DEVICES
# ----------------------------
def worker_process(rank: int, gpu_id: int, model_root: str, image_infos: List[dict], categories_map: dict):
    """
    rank: worker index
    gpu_id: which GPU this worker should use (absolute id)
    model_root: absolute path to folder containing model files (config.json, *.safetensors, tokenizer, etc.)
    image_infos: list of image dicts assigned to this worker
    categories_map: {label_name -> category_id}
    """
    # Ensure separate env for each process so transformers/accelerate sees only one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Optionally reduce parallel CPU threads to avoid oversubscription
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"

    # Import heavy libraries here
    import torch
    from PIL import Image
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"[worker {rank}] starting on CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    # Determine dtype
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    # Load model + processor (device_map='auto' will map to the single visible GPU)
    print(f"[worker {rank}] loading model from: {model_root}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_root,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_root, trust_remote_code=True)
    
    # Set padding side to left for decoder-only architecture
    processor.tokenizer.padding_side = 'left'

    model.eval()
    if use_cuda:
        # verify current device
        dev = torch.device("cuda")
        print(f"[worker {rank}] model devices: {set(p.device for p in model.parameters() if p.device is not None)}")
    else:
        print(f"[worker {rank}] running on CPU (no CUDA)")

    partial_results = []

    # Process in batches
    batch_imgs = []
    batch_img_ids = []
    
    if os.environ.get("QWEN_VL_PLOT", "0") == "1":
        os.makedirs(VIS_DIR, exist_ok=True)

    batch_index = 0
    
    # Progress tracking
    total_imgs = len(image_infos)
    start_time = time.time()
    processed = 0
    
    for entry in image_infos:
        img_id = entry["id"]
        file_name = entry["file_name"]
        img_path = os.path.join(IMAGE_DIR, file_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[worker {rank}] warning: cannot open {img_path}: {e}")
            continue

        batch_imgs.append(img)
        batch_img_ids.append(img_id)

        if len(batch_imgs) >= BATCH_SIZE:
            _run_batch(batch_imgs, batch_img_ids, processor, model, categories_map, partial_results, batch_index=batch_index, rank=rank)
            batch_imgs, batch_img_ids = [], []
            batch_index += 1

        # Update progress
        processed += 1
        if processed % 20 == 0:  # update every 20 images
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = total_imgs - processed
            eta = remaining / rate if rate > 0 else 0
            print(f"[worker {rank}] Progress: {processed}/{total_imgs} "
                  f"({processed/total_imgs*100:.2f}%) ETA {eta/60:.1f} min")

    # final partial batch
    if batch_imgs:
        _run_batch(batch_imgs, batch_img_ids, processor, model, categories_map, partial_results, batch_index=batch_index, rank=rank)

    # Save partial results
    os.makedirs(PARTIAL_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(PARTIAL_RESULTS_DIR, f"qwen_partial_results_{rank}.json")
    with open(out_path, "w") as f:
        json.dump(partial_results, f)
    print(f"[worker {rank}] wrote {len(partial_results)} detections to {out_path}")


# ----------------------------
# Batch runner used by worker (kept outside to reduce code duplication)
# ----------------------------
def _run_batch(batch_imgs, batch_img_ids, processor, model, categories_map, partial_results, batch_index=None, rank=None):
    """
    Given list of PIL images and corresponding image_ids, run the model and append COCO-format results to partial_results.
    """
    import torch
    from qwen_vl_utils import process_vision_info
    import re

    # Build messages using the improved template
    messages = [ 
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": (
                        "Detect all objects in the image that belong to the following categories: " + ", ".join(COCO_CLASSES) + "."
                        "Return ONLY a single valid JSON object with this exact schema (and nothing else, no markdown, no code fences): "
                        "{\"objects\":[{\"label\":\"<coco-class>\",\"box\":[x1,y1,x2,y2],\"confidence\":<float-num>}]}. "
                        "Do not wrap the JSON in backticks. Do not output a list at the top-level. "
                        "Use integer coordinates for box as [x1,y1,x2,y2]. If unsure, return {\"objects\":[]}. "
                        "Use only class names provided in the list above."
                    )},
                ],
            }
        ]
        for img in batch_imgs
    ]

    # Prepare input texts and vision tensors using qwen_vl_utils helpers
    texts = []
    image_inputs = []
    for msg in messages:
        text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        t_img, _ = process_vision_info(msg)
        texts.append(text)
        image_inputs.append(t_img)

    # Prepare tensors
    device = next(model.parameters()).device
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Move inputs to device carefully
    # processor may produce pixel_values and other tensors nested in dicts - use to(device)
    try:
        inputs = inputs.to(device)
    except Exception:
        # Fallback: move tensors individually
        for k, v in list(inputs.items()):
            try:
                inputs[k] = v.to(device)
            except Exception:
                pass

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            top_p=1.0,
        )

    # Trim already-present input ids and decode outputs
    gen_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated)
    ]

    decoded_texts = processor.batch_decode(gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        # ```json ... ``` or ``` ... ```
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1).strip()
        return s

    def _find_first_json_block(s: str) -> str | None:
        """
        Extract the first top-level JSON object or array string.
        """
        s = s.strip()
        start_positions = [s.find("{"), s.find("[")]
        start_positions = [p for p in start_positions if p != -1]
        if not start_positions:
            return None
        start = min(start_positions)
        stack = []
        for i, ch in enumerate(s[start:], start=start):
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    return None
                opening = stack.pop()
                if (opening == "{" and ch != "}") or (opening == "[" and ch != "]"):
                    return None
                if not stack:
                    # closed the top-level block
                    return s[start:i+1]
        return None

    def _parse_any_json(s: str):
        """
        Parse JSON from a potentially messy string:
        - remove code fences
        - try full parse
        - else find first top-level JSON block and parse
        """
        s0 = _strip_code_fences(s)
        try:
            return json.loads(s0)
        except Exception:
            pass
        block = _find_first_json_block(s0)
        if block is None:
            return None
        try:
            return json.loads(block)
        except Exception:
            return None

    def _coerce_to_objects(obj) -> list:
        """
        Accept either:
        - {"objects":[...]}
        - {"detections":[...]}
        - [{"label":..., "box":...}, ...]
        - {"label":..., "box":...}
        """
        if obj is None:
            return []
        if isinstance(obj, dict):
            if "objects" in obj and isinstance(obj["objects"], list):
                return obj["objects"]
            if "detections" in obj and isinstance(obj["detections"], list):
                return obj["detections"]
            # Single detection as dict
            if "label" in obj and ("box" in obj or "bbox" in obj or "bbox_2d" in obj):
                return [obj]
            return []
        if isinstance(obj, list):
            return obj
        return []

    def _normalize_label(text: str) -> str:
        l = (text or "").strip().lower()
        # quick common mappings to COCO 'person'
        person_triggers = ["player", "skier", "man", "woman", "boy", "girl", "people", "person"]
        if any(t in l for t in person_triggers):
            return "person"
        return l

    def _map_label_to_coco(label_text: str, categories_map: dict) -> str | None:
        """
        Try exact match, then substring longest match among official COCO names.
        """
        if not label_text:
            return None
        l = _normalize_label(label_text)
        if l in categories_map:
            return l
        # longest substring match of category name within label
        candidates = [name for name in categories_map.keys() if name in l]
        if candidates:
            return max(candidates, key=len)
        return None

    def _extract_box(det) -> list | None:
        """
        Accept various bbox encodings:
        - 'box': [x1,y1,x2,y2] or {'x1':..,'y1':..,'x2':..,'y2':..}
        - 'bbox' / 'bbox_2d': same shapes as above
        - 'xyxy': [x1,y1,x2,y2]
        """
        val = None
        for k in ("box", "bbox", "bbox_2d", "xyxy"):
            if k in det:
                val = det[k]
                break
        # dict form
        if isinstance(val, dict):
            keys = {kk.lower(): vv for kk, vv in val.items()}
            needed = ["x1", "y1", "x2", "y2"]
            if all(k in keys for k in needed):
                return [float(keys["x1"]), float(keys["y1"]), float(keys["x2"]), float(keys["y2"])]
        # list/tuple form
        if isinstance(val, (list, tuple)) and len(val) == 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in val]
                return [x1, y1, x2, y2]
            except Exception:
                return None
        return None

    # Parse each output (expect JSON string)
    for decoded, img_id in zip(decoded_texts, batch_img_ids):
        parsed = _parse_any_json(decoded)
        detections = _coerce_to_objects(parsed)
        if not detections:
            continue

        for det in detections:
            if not isinstance(det, dict):
                continue
            label_raw = det.get("label") or det.get("class") or det.get("category") or det.get("name")
            label_mapped = _map_label_to_coco(label_raw, categories_map)
            if not label_mapped:
                continue
            box = _extract_box(det)
            if not box:
                continue
            x1, y1, x2, y2 = box
            # fix inverted boxes if any
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            w = x2 - x1
            h = y2 - y1
            if w <= 1 or h <= 1:
                continue
            score = det.get("confidence") or det.get("score") or det.get("prob") or 0.5
            try:
                score = float(score)
            except Exception:
                score = 0.5
            partial_results.append({
                "image_id": img_id,
                "category_id": categories_map[label_mapped],
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": score,
            })
            

    # Visualization
    if batch_index is not None and os.environ.get("QWEN_VL_PLOT", "0") == "1":
        try:
            _visualize_batch(
                batch_imgs[0],
                decoded_texts[0],
                os.path.join(VIS_DIR, f"worker_{rank}_batch_{batch_index}.jpg")
            )
        except Exception as e:
            print(f"[worker {rank}] visualization failed: {e}")


# ----------------------------
# Parent orchestrator
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL COCO few-shot detection")
    parser.add_argument("--plot", action="store_true", default=False, help="Enable saving visualization images during inference")
    args = parser.parse_args()

    # Control plotting via env to pass to workers
    os.environ["QWEN_VL_PLOT"] = "1" if args.plot else "0"

    # 1) find model snapshot dir
    try:
        model_snapshot = find_snapshot_dir(HF_CACHE_BASE)
    except FileNotFoundError as e:
        print("ERROR: model snapshot not found in the provided HF cache base.")
        print("You can either run huggingface snapshot_download(...) or point HF_CACHE_BASE to the folder containing snapshots.")
        raise

    print("Using model snapshot dir:", model_snapshot)

    # 2) load COCO metadata (light)
    with open(ANNOTATIONS, "r") as f:
        coco_meta = json.load(f)
    categories = {c["name"]: c["id"] for c in coco_meta["categories"]}
    images_info = coco_meta["images"]

    # If results already exist, skip inference and only run evaluation
    if os.path.exists(OUTPUT_JSON):
        print(f"Found existing results at {OUTPUT_JSON}. Skipping inference and running evaluation only.")
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            # Load full GT and filter to few-shot classes similar to coco_ref_dataset
            coco_full = COCO(ANNOTATIONS)
            cat_ids = coco_full.getCatIds(catNms=COCO_CLASSES)
            ann_ids = coco_full.getAnnIds(catIds=cat_ids)
            filtered_anns = coco_full.loadAnns(ann_ids)
            coco_gt = COCO()
            coco_gt.dataset = coco_full.dataset.copy()
            coco_gt.dataset['annotations'] = filtered_anns
            coco_gt.createIndex()
            # Load results that may be dataset-like or a plain list
            with open(OUTPUT_JSON, "r") as rf:
                res_data = json.load(rf)
            if isinstance(res_data, dict) and "annotations" in res_data:
                res_list = res_data["annotations"]
            elif isinstance(res_data, list):
                res_list = res_data
            else:
                raise ValueError("Unsupported results JSON format. Expect list or dict with 'annotations'.")
            coco_dt = coco_gt.loadRes(res_list)
            evaluator = COCOeval(coco_gt, coco_dt, "bbox")
            evaluator.params.imgIds = list(coco_gt.imgs.keys())
            evaluator.params.catIds = cat_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        except Exception as e:
            print("COCO evaluation failed:", e)
            print("You can run COCO eval manually using pycocotools.")
        return

    # 3) determine GPUs available
    # Use torch to detect CUDA_VISIBLE_DEVICES effect in this parent process
    try:
        import torch
        total_gpus = torch.cuda.device_count()
    except Exception:
        total_gpus = 0

    if total_gpus <= 0:
        print("No GPUs detected. This script is intended to run with GPUs. Will run single process on CPU.")
        total_gpus = 1

    print(f"Detected GPUs: {total_gpus}")

    # 4) split image list across workers - round robin for balanced distribution
    per_worker_lists = [[] for _ in range(total_gpus)]
    for idx, info in enumerate(images_info):
        per_worker_lists[idx % total_gpus].append(info)

    # 5) spawn worker processes
    os.makedirs(PARTIAL_RESULTS_DIR, exist_ok=True)
    processes = []
    for rank in range(total_gpus):
        gpu_id = rank  # absolute GPU id; we set CUDA_VISIBLE_DEVICES in worker to this id
        p = mp.Process(
            target=worker_process,
            args=(rank, gpu_id, model_snapshot, per_worker_lists[rank], categories),
            daemon=False,
        )
        p.start()
        processes.append(p)
        time.sleep(1.0)  # small stagger

    # 6) wait for workers
    for p in processes:
        p.join()

    # 7) collect partial results and merge
    merged = []
    for fn in sorted(Path(PARTIAL_RESULTS_DIR).glob("qwen_partial_results_*.json")):
        with open(fn, "r") as f:
            try:
                data = json.load(f)
                merged.extend(data)
            except Exception:
                print(f"Warning: couldn't read {fn}")

    # Build COCO-style dataset result:
    # - images: subset of GT images that appear in predictions
    # - annotations: predictions as annotation-like entries with ids
    # - categories: only few-shot classes (matching GT ids and names)
    img_by_id = {img["id"]: img for img in images_info}
    used_img_ids = sorted({d["image_id"] for d in merged})
    images_out = [img_by_id[i] for i in used_img_ids if i in img_by_id]

    # Filter categories to few-shot list while preserving id/name/supercategory from GT
    valid_names = set(COCO_CLASSES)
    categories_out = [c for c in coco_meta["categories"] if c["name"] in valid_names]

    annotations_out = []
    next_ann_id = 1
    for det in merged:
        x, y, w, h = det["bbox"]
        ann = {
            "id": next_ann_id,
            "image_id": det["image_id"],
            "category_id": det["category_id"],
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(max(w, 0.0) * max(h, 0.0)),
            "iscrowd": 0,
            "score": float(det.get("score", 1.0)),
        }
        annotations_out.append(ann)
        next_ann_id += 1

    results_dataset = {
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories_out,
    }

    # Save merged results in dataset-like JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_dataset, f)
    print(f"Merged results saved to {OUTPUT_JSON} (images={len(images_out)}, annotations={len(annotations_out)})")

    # 8) Evaluate with COCO on few-shot classes
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_full = COCO(ANNOTATIONS)
        cat_ids = coco_full.getCatIds(catNms=COCO_CLASSES)
        ann_ids = coco_full.getAnnIds(catIds=cat_ids)
        filtered_anns = coco_full.loadAnns(ann_ids)
        coco_gt = COCO()
        coco_gt.dataset = coco_full.dataset.copy()
        coco_gt.dataset['annotations'] = filtered_anns
        coco_gt.createIndex()
        # Use in-memory annotations_out directly for results
        coco_dt = coco_gt.loadRes(annotations_out)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = list(coco_gt.imgs.keys())
        evaluator.params.catIds = cat_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    except Exception as e:
        print("COCO evaluation failed:", e)
        print("You can run COCO eval manually using pycocotools.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
