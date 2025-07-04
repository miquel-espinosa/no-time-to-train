#!/bin/bash

# Run ablation on the results std for COCO few-shot results

SEEDS="42 13 27 36 88 33 69 55 77 99"
SHOTS="1 2 3 5 10 30"
CONFIG=./dev_hongyi/new_exps/coco_fewshot_10shot_Sam2L.yaml
GPUS=4

for SEED in $SEEDS; do
    for SHOT in $SHOTS; do
        echo "=====> Running few-shot pipeline for $SHOT shot with seed $SEED"
        CUDA_VISIBLE_DEVICES=1,2,3,4 zsh ./few_shot_full_pipeline.sh $CONFIG $SHOT $GPUS $SEED
    done
done