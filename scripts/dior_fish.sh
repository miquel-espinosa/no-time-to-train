#! /bin/bash

echo "Running DIOR"

# SHOTS=(1 5 10)
SHOTS=(1)

# DATASET_NAME=DIOR
# DEVICES=5,6
# CAT_NAMES='Expressway-Service-area,Expressway-toll-station,airplane,airport,baseballfield,basketballcourt,bridge,chimney,dam,golffield,groundtrackfield,harbor,overpass,ship,stadium,storagetank,tenniscourt,trainstation,vehicle,windmill'
# CATEGORY_NUM=20

# for SHOT in "${SHOTS[@]}"; do  
#     zsh scripts/matching_cdfsod_pipeline.sh "$DATASET_NAME" "$SHOT" "$DEVICES" "$CAT_NAMES" "$CATEGORY_NUM"
# done


echo "Running FISH"


DATASET_NAME=FISH
DEVICES=5,6
CAT_NAMES='fish'
CATEGORY_NUM=1

for SHOT in "${SHOTS[@]}"; do
    zsh scripts/matching_cdfsod_pipeline.sh "$DATASET_NAME" "$SHOT" "$DEVICES" "$CAT_NAMES" "$CATEGORY_NUM"
done
