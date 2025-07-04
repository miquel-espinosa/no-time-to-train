#!/bin/bash

export DEVICES=$1

# Run SAM segmentation for PASCAL VOC
run_sam_segmentation() {
    local input_json=$1
    local image_dir=$2
    local visualize=$3
    local output_json="${input_json%.json}_with_segm.json"
    
    if [ -f "$output_json" ]; then # Check if output file already exists
        echo "Output file $output_json already exists. Skipping segmentation."
        return
    fi
    
    echo "Running SAM segmentation for $input_json"
    CUDA_VISIBLE_DEVICES=$DEVICES python dev_hongyi/dataset/sam_bbox_to_segm_batch.py \
        --input_json "$input_json" \
        --image_dir "$image_dir" \
        --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
        --model_type vit_h \
        --device cuda \
        --batch_size 8 \
        $([ "$visualize" = true ] && echo "--visualize")
}

# Since PASCAL VOC does not provide instance-segmentation masks, we will use the
# bounding boxes as prompt for SAM to generate masks, and save them as a new json file
# trainval json file
run_sam_segmentation ./data/pascal_voc/annotations/voc0712_trainval.json \
                     ./data/pascal_voc/VOCdevkit/allimages \
                     false
# test json file
run_sam_segmentation ./data/pascal_voc/annotations/voc07_test.json \
                     ./data/pascal_voc/VOCdevkit/allimages \
                     false