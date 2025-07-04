#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

python run_lightening.py test --config $CONFIG --model.test_mode vis_memory --ckpt_path ./tmp_ckpts/1111.ckpt --trainer.devices $GPUS
