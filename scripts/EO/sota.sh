#! /bin/bash

# Parse arguments
SHOTS=(1 2 3 5 10)
DEVICES=0,1,2,3,4,5,6,7
MODELS=(dinov3_l dinov3_sat_l dinov2_l)

DATASET_NAME=SOTA
SEED=42
VIS_THR=0.6
MEMORY_VIS=true
PREDICTION_VIS=true
CLEAN_CKPTS=true
HEURISTICS=false
FORCE=false

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
            $([ "$CLEAN_CKPTS" = true ] && echo "--clean-ckpts") \
            $([ "$HEURISTICS" = true ] && echo "--heuristics") \
            $([ "$FORCE" = true ] && echo "--force")
        echo "EO for $DATASET_NAME with $SHOT shots and $MODEL model completed"
    done
done
