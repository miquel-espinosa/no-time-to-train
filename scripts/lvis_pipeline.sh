#!/usr/bin/env bash

SPLIT=$1
SHOTS=$2
GPUS=$3

# Check that split is one of the valid splits
if [ $SPLIT != "lvis" ] && [ $SPLIT != "lvis_common" ] && [ $SPLIT != "lvis_frequent" ] && \
   [ $SPLIT != "lvis_rare" ] && [ $SPLIT != "lvis_minival" ] && [ $SPLIT != "lvis_minival_common" ] && \
   [ $SPLIT != "lvis_minival_frequent" ] && [ $SPLIT != "lvis_minival_rare" ]; then
    echo "Error: Split must be one of:"
    echo "  lvis"
    echo "  lvis_common"
    echo "  lvis_frequent" 
    echo "  lvis_rare"
    echo "  lvis_minival"
    echo "  lvis_minival_common"
    echo "  lvis_minival_frequent"
    echo "  lvis_minival_rare"
    exit 1
fi

SEED=42
YAML_FILE=./dev_hongyi/pl_configs/matching_lvis_allClass.yaml
FILENAME=few_shot_lvis_ann_${SHOTS}shot_seed${SEED}_fixed.pkl

RESULTS_DIR=./work_dirs/${SPLIT}_seed${SEED}/${SHOTS}shot
mkdir -p $RESULTS_DIR
# Make sure the lvis_results directory exists, otherwise print error and exit
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: $RESULTS_DIR directory does not exist. Create a symbolic link to the results directory."
    exit 1
fi

# Set the number of categories for the split
if [ "$SPLIT" = "lvis" ] || [ "$SPLIT" = "lvis_minival" ]; then
    CAT_NUM=1203
elif [ "$SPLIT" = "lvis_common" ] || [ "$SPLIT" = "lvis_minival_common" ]; then
    CAT_NUM=461
elif [ "$SPLIT" = "lvis_frequent" ] || [ "$SPLIT" = "lvis_minival_frequent" ]; then
    CAT_NUM=405
elif [ "$SPLIT" = "lvis_rare" ] || [ "$SPLIT" = "lvis_minival_rare" ]; then
    CAT_NUM=337
else
    echo "Error: Invalid split: $SPLIT"
    exit 1
fi

REFERENCES_JSON_FILE=./data/lvis/lvis_v1_train.json # Always the same (from the train set, though there is no training)
if [ "$SPLIT" = "lvis_minival" ] || [ "$SPLIT" = "lvis_minival_common" ] || [ "$SPLIT" = "lvis_minival_frequent" ] || [ "$SPLIT" = "lvis_minival_rare" ]; then
    TEST_JSON_FILE=./data/lvis/lvis_v1_minival.json
elif [ "$SPLIT" = "lvis" ] || [ "$SPLIT" = "lvis_common" ] || [ "$SPLIT" = "lvis_frequent" ] || [ "$SPLIT" = "lvis_rare" ]; then
    TEST_JSON_FILE=./data/lvis/lvis_v1_val.json
else
    echo "Error: Invalid split: $SPLIT"
    exit 1
fi

# Generated file will have the format of <out-path>_<n_shot>shot_seed<seed>.pkl
python dev_hongyi/dataset/few_shot_sampling.py --n-shot $SHOTS --out-path ${RESULTS_DIR}/${FILENAME} --seed $SEED \
                                               --dataset lvis --plot --img-dir ./data/coco/allimages

python run_lightening.py test --config $YAML_FILE \
                              --model.test_mode fill_memory \
                              --out_path ${RESULTS_DIR}/memory.ckpt \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.model_cfg.memory_bank_cfg.category_num $CAT_NUM \
                              --model.init_args.dataset_cfgs.fill_memory.memory_pkl ${RESULTS_DIR}/${FILENAME} \
                              --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOTS \
                              --model.init_args.dataset_cfgs.fill_memory.class_split $SPLIT \
                              --model.init_args.dataset_cfgs.fill_memory.json_file $REFERENCES_JSON_FILE \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS

python run_lightening.py test --config $YAML_FILE \
                              --model.test_mode postprocess_memory \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.model_cfg.memory_bank_cfg.category_num $CAT_NUM \
                              --ckpt_path ${RESULTS_DIR}/memory.ckpt \
                              --out_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices 1

python run_lightening.py test --config $YAML_FILE \
                              --ckpt_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                              --model.init_args.test_mode test \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.model_cfg.memory_bank_cfg.category_num $CAT_NUM \
                              --model.init_args.dataset_cfgs.test.class_split $SPLIT \
                              --model.init_args.dataset_cfgs.test.json_file $TEST_JSON_FILE \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS