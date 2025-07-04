#! /bin/bash

## LAUNCH COMMAND EXAMPLE:
## CUDA_VISIBLE_DEVICES=0,1 ./scripts/pascal_split.sh 1 2  # split 1, using 2 GPUs

# PASCA VOC split to evaluate on. Either 1, 2, or 3
PASCAL_SPLIT=$1
# GPUS is the number of GPUs to use
GPUS=$2
# SHOTS can be either a single number (e.g. 1) or multiple nubmers (e.g. 1,2,3)
SHOTS="1 2 3 5 10"
# SHOTS="3 5 10" # NOTE: temporary because we have already run 1, 2 shot for split 1
# SEED is the seed to use for the random number generator
SEED=42
# YAML file to use for the config
YAML_FILE=dev_hongyi/pl_configs/matching_pascal_voc_few_shot_template.yaml

# Novel classes are always 5
CATEGORY_NUM=5
# if pascal split is 1, then CAT_NAMES is
if [ $PASCAL_SPLIT -eq 1 ]; then # ['bus', 'sofa', 'cow', 'bird', 'motorbike']
    CAT_NAMES='bus,sofa,cow,bird,motorbike'
elif [ $PASCAL_SPLIT -eq 2 ]; then # ['horse', 'aeroplane', 'sofa', 'cow', 'bottle']
    CAT_NAMES='horse,aeroplane,sofa,cow,bottle'
elif [ $PASCAL_SPLIT -eq 3 ]; then # ['sheep', 'sofa', 'boat', 'cat', 'motorbike']
    CAT_NAMES='sheep,sofa,boat,cat,motorbike'
fi

for SHOT in $SHOTS; do
    # Echo in red color "We are starting pipeline for $SHOT shot"
    echo -e "\033[31mPASCAL SPLIT $PASCAL_SPLIT, SHOT $SHOT\033[0m"

    RESULTS_DIR=work_dirs/pascal_voc_split_${PASCAL_SPLIT}_seed${SEED}/${SHOT}shot
    # If the results directory does not exist, create it
    if [ ! -d "$RESULTS_DIR" ]; then
        mkdir -p $RESULTS_DIR
    fi
    FILENAME=few_shot_ann_${SHOT}shot_seed${SEED}_fixed.pkl

    mkdir -p $RESULTS_DIR


    # Generated file will have the format of <out-path>_<n_shot>shot_seed<seed>.pkl
    python dev_hongyi/dataset/few_shot_sampling.py --n-shot $SHOT \
                                                --out-path $RESULTS_DIR/${FILENAME} \
                                                --seed $SEED \
                                                --dataset pascal_voc_split_${PASCAL_SPLIT} \
                                                --img-dir ./data/pascal_voc/VOCdevkit/allimages \
                                                --plot

    python run_lightening.py test --config $YAML_FILE \
                                --model.test_mode fill_memory \
                                --out_path $RESULTS_DIR/memory.ckpt \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --model.init_args.dataset_cfgs.fill_memory.memory_pkl $RESULTS_DIR/${FILENAME} \
                                --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
                                --model.init_args.dataset_cfgs.fill_memory.cat_names $CAT_NAMES \
                                --model.init_args.dataset_cfgs.fill_memory.class_split pascal_voc_split_$PASCAL_SPLIT \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices $GPUS

    python run_lightening.py test --config $YAML_FILE \
                                --model.test_mode postprocess_memory \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --ckpt_path $RESULTS_DIR/memory.ckpt \
                                --out_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices 1

    python run_lightening.py test --config $YAML_FILE  \
                                --ckpt_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --model.init_args.test_mode test \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --model.init_args.dataset_cfgs.test.cat_names $CAT_NAMES \
                                --model.init_args.dataset_cfgs.test.class_split pascal_voc_split_$PASCAL_SPLIT \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices $GPUS
done