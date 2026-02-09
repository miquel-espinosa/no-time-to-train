#! /bin/bash

# Parse arguments
SHOTS=(1 2 3 5 10)
DEVICES=6,7
MODELS=(dinov3_l dinov3_sat_l dinov2_l)

DATASET_NAME=iSAID
SEED=42
VIS_THR=0.4
MEMORY_VIS=true
PREDICTION_VIS=false
CLEAN_CKPTS=true

for SHOT in "${SHOTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "Running EO for $DATASET_NAME with $SHOT shots and $MODEL model"
        ./scripts/EO/EO_template.sh \
            --dataset $DATASET_NAME \
            --shot $SHOT \
            --model $MODEL \
            --seed $SEED \
            --devices $DEVICES \
            --vis-thr $VIS_THR \
            $([ "$MEMORY_VIS" = true ] && echo "--memory-vis") \
            $([ "$PREDICTION_VIS" = true ] && echo "--prediction-vis") \
            $([ "$CLEAN_CKPTS" = true ] && echo "--clean-ckpts")
        echo "EO for $DATASET_NAME with $SHOT shots and $MODEL model completed"
    done
done
