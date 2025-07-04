#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

#echo "\n>>> Step 1: Filling memory with references"
#python run_lightening.py test --config $CONFIG --model.test_mode fill_memory --out_path ./tmp_ckpts/0000.ckpt --trainer.devices $GPUS
#
#echo "\n\n>>> Step 2: Post-processing memories"
#python run_lightening.py test --config $CONFIG --model.test_mode postprocess_memory --ckpt_path ./tmp_ckpts/0000.ckpt --out_path ./tmp_ckpts/1111.ckpt --trainer.devices 1
#
#echo "\n\n>>> Step 3: Testing on support set"
#python run_lightening.py test --config $CONFIG --model.test_mode test_support --ckpt_path ./tmp_ckpts/1111.ckpt --out_support_res ./tmp_ckpts/support_res.pkl  --trainer.devices $GPUS
#
#echo "\n\n>>> Step 4: Sampling negative references"
#python sample_negative_offline.py --config $CONFIG --out_neg_pkl ./tmp_ckpts/sampled_neg.pkl --out_neg_json ./tmp_ckpts/sampled_neg.json --out_support_res ./tmp_ckpts/support_res.pkl
##
#echo "\n\n>>> Step 5: Filling negative memory"
#python run_lightening.py test --config $CONFIG --model.test_mode fill_memory_neg --ckpt_path ./tmp_ckpts/1111.ckpt --out_path ./tmp_ckpts/2222.ckpt --out_neg_pkl ./tmp_ckpts/sampled_neg.pkl --out_neg_json ./tmp_ckpts/sampled_neg.json --trainer.devices $GPUS
#
#echo "\n\n>>> Step 6: Post-processing negative memories"
#python run_lightening.py test --config $CONFIG --model.test_mode postprocess_memory_neg --ckpt_path ./tmp_ckpts/2222.ckpt --out_path ./tmp_ckpts/3333.ckpt --trainer.devices 1

echo "\n\n>>> Step 7: Testing on the target set"
python run_lightening.py test --config $CONFIG --model.test_mode test --ckpt_path ./tmp_ckpts/3333.ckpt --trainer.devices $GPUS