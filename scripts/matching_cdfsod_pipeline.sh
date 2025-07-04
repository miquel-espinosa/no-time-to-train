#! /bin/bash

# Parse arguments
DATASET_NAME=$1
SHOT=$2
DEVICES=$3
CAT_NAMES=$4
CATEGORY_NUM=$5

# Additional settings
NUM_DEVICES=$((1+${#DEVICES//[^,]/})) # Count the number of commas in DEVICES
ALL_DATASETS_PATH=/localdisk/data2/Users/s2254242/datasets
DATASET_PATH=$ALL_DATASETS_PATH/$DATASET_NAME
YAML_PATH=dev_hongyi/pl_configs/matching_cdfsod_template.yaml
PATH_TO_SAVE_CKPTS=./tmp_ckpts/cd_fsod/matching
mkdir -p $PATH_TO_SAVE_CKPTS
FIRST_DEVICE=${DEVICES%%,*}

# HELPER FUNCTIONS
# Function to run SAM segmentation
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

# FIRST STEP: SAM-H to segment the dataset using the bounding boxes
# --------------------
# run_sam_segmentation $DATASET_PATH/annotations/$SHOT\_shot.json $DATASET_PATH/train true
# run_sam_segmentation $DATASET_PATH/annotations/train.json $DATASET_PATH/train
# run_sam_segmentation $DATASET_PATH/annotations/test.json $DATASET_PATH/test false

# SECOND STEP: convert the COCO annotations to a pickle file
# --------------------
# Usage: python script.py <input_json_path> <output_pkl_path>
# python dev_hongyi/dataset/coco_to_pkl.py \
#     $DATASET_PATH/annotations/$SHOT\_shot\_with_segm.json \
#     $DATASET_PATH/annotations/$SHOT\_shot\_with_segm.pkl \
#     $SHOT
# echo "Second step done: converting to pickle"

# THIRD STEP: Fill memory with references
# --------------------
CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
    --model.test_mode fill_memory \
    --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
    --model.init_args.dataset_cfgs.fill_memory.root $DATASET_PATH/train \
    --model.init_args.dataset_cfgs.fill_memory.json_file $DATASET_PATH/annotations/$SHOT\_shot\_with_segm.json \
    --model.init_args.dataset_cfgs.fill_memory.memory_pkl $DATASET_PATH/annotations/$SHOT\_shot\_with_segm.pkl \
    --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
    --model.init_args.dataset_cfgs.fill_memory.cat_names $CAT_NAMES \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1

echo "Third step done: Memory filled"

# FOURTH STEP: Postprocess memories, e.g., computer averages and run kmeans
# --------------------
CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
    --model.test_mode postprocess_memory \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
    --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1
echo "Fourth step done: Postprocessing memories"

# FIFTH STEP: Testing on the target set
# --------------------
echo -e "\033[31mEVALUATING $SHOT SHOT FOR DATASET $DATASET_NAME\033[0m"
CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
    --model.test_mode test \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --model.init_args.dataset_cfgs.test.root $DATASET_PATH/test \
    --model.init_args.dataset_cfgs.test.json_file $DATASET_PATH/annotations/test\_with_segm.json \
    --model.init_args.dataset_cfgs.test.cat_names $CAT_NAMES \
    --trainer.devices $NUM_DEVICES
echo "Fifth step done: Testing on the target set"