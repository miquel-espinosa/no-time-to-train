#! /bin/bash

echo "Running ArTaxOr"

# SHOTS=(1 5 10)
SHOTS=(1)

# DATASET_NAME=ArTaxOr
# DEVICES=4
# CAT_NAMES='Araneae,Coleoptera,Diptera,Hemiptera,Hymenoptera,Lepidoptera,Odonata'
# CATEGORY_NUM=7

# for SHOT in "${SHOTS[@]}"; do
#     zsh scripts/matching_cdfsod_pipeline.sh "$DATASET_NAME" "$SHOT" "$DEVICES" "$CAT_NAMES" "$CATEGORY_NUM"
# done


echo "Running clipart1k"


DATASET_NAME=clipart1k
DEVICES=5
CAT_NAMES='sheep,chair,boat,bottle,diningtable,sofa,cow,motorbike,car,aeroplane,cat,train,person,bicycle,pottedplant,bird,dog,bus,tvmonitor,horse'
CATEGORY_NUM=20

for SHOT in "${SHOTS[@]}"; do
    zsh scripts/matching_cdfsod_pipeline.sh "$DATASET_NAME" "$SHOT" "$DEVICES" "$CAT_NAMES" "$CATEGORY_NUM"
done
