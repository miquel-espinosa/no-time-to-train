#! /bin/bash

echo "Running NEU-DET"

# SHOTS=(1 5 10)
SHOTS=(1)

DATASET_NAME=NEU-DET
DEVICES=4,5
CAT_NAMES='crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches'
CATEGORY_NUM=6

for SHOT in "${SHOTS[@]}"; do
    zsh scripts/matching_cdfsod_pipeline.sh "$DATASET_NAME" "$SHOT" "$DEVICES" "$CAT_NAMES" "$CATEGORY_NUM"
done


echo "Running UODD"


DATASET_NAME=UODD
DEVICES=4,5
CAT_NAMES='seacucumber,seaurchin,scallop'
CATEGORY_NUM=3

for SHOT in "${SHOTS[@]}"; do
    zsh scripts/matching_cdfsod_pipeline.sh "$DATASET_NAME" "$SHOT" "$DEVICES" "$CAT_NAMES" "$CATEGORY_NUM"
done
