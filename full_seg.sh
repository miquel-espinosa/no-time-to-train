#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

# Fill memory with references
python run_lightening.py test --config $CONFIG --model.test_mode fill_memory --out_path ./tmp_ckpts/0000.ckpt --trainer.devices $GPUS

#  Postprocess memories, e.g., computer averages and run kmeans
python run_lightening.py test --config $CONFIG --model.test_mode postprocess_memory --ckpt_path ./tmp_ckpts/0000.ckpt --out_path ./tmp_ckpts/1111.ckpt --trainer.devices 1

# testing on the target set
python run_lightening.py test --config $CONFIG --model.test_mode test --ckpt_path ./tmp_ckpts/1111.ckpt --trainer.devices $GPUS