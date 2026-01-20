# CLASSES=("dining_table")
CLASSES=("potted_plant")

for CLASS in "${CLASSES[@]}"; do

    INSTANCES=(data/coco/annotations/per_class_instances/$CLASS/instances/${CLASS}_*.json)

    for INSTANCE in "${INSTANCES[@]}"; do

        echo "Running pipeline for $CLASS $INSTANCE"
        
        CUDA_VISIBLE_DEVICES=7 ./scripts/1shot_ref_ablation/template.sh $INSTANCE $CLASS 33 true

    done

done


