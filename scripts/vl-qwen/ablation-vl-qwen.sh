
# STEP 1

# Run VL-QWEN to detect few-shot classes in validation images.
# Outputs are stored in json file
python3 vlm/run_qwen_coco_detections.py 

# Use --plot to save visualization images during inference
# python3 vlm/run_qwen_coco_detections.py --plot


# STEP 2

# Convert bbox results to instance segmentation results with SAM
python no_time_to_train/dataset/sam_bbox_to_segm_batch.py \
    --input_json vlm/qwen_outputs/qwen_coco_results.json \
    --image_dir data/coco/val2017 \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --model_type vit_h \
    --device cuda \
    --batch_size 4


# STEP 3 

# Evaluate final results
python3 vlm/evaluate_results.py \
    --gt ./data/coco/annotations/instances_val2017.json \
    --pred vlm/qwen_outputs/qwen_coco_results_with_segm.json