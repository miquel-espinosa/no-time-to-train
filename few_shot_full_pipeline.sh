#!/usr/bin/env bash

CONFIG=$1
SHOTS=$2
GPUS=$3
SEED=$4

RESULTS_DIR=work_dirs/few_shot_results
mkdir -p $RESULTS_DIR

FILENAME=few_shot_ann_${SHOTS}shot_seed${SEED}_fixed.pkl

# Generated file will have the format of <out-path>_<n_shot>shot_seed<seed>.pkl
python dev_hongyi/dataset/few_shot_sampling.py --n-shot $SHOTS --out-path ${RESULTS_DIR}/${FILENAME} --seed $SEED

python run_lightening.py test --config $CONFIG \
                              --model.test_mode fill_memory \
                              --out_path ${RESULTS_DIR}/memory.ckpt \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --model.init_args.dataset_cfgs.fill_memory.memory_pkl ${RESULTS_DIR}/${FILENAME} \
                              --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOTS \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS

python run_lightening.py test --config $CONFIG \
                              --model.test_mode postprocess_memory \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --ckpt_path ${RESULTS_DIR}/memory.ckpt \
                              --out_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                              --trainer.devices 1

python run_lightening.py test --config $CONFIG  \
                              --ckpt_path ${RESULTS_DIR}/memory_postprocessed.ckpt \
                              --model.init_args.test_mode test \
                              --model.init_args.model_cfg.memory_bank_cfg.length $SHOTS \
                              --trainer.logger.save_dir ${RESULTS_DIR}/ \
                              --trainer.devices $GPUS