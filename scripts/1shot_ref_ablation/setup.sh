# Approx total run time ~4min

# First create per class json files
# python no_time_to_train/dataset/create_jsons_per_class.py \
#   --input_dir data/coco/annotations \
#   --output_dir data/coco/annotations/per_class_instances \
#   --train_file instances_train2017.json \
#   --val_file instances_val2017.json \
#   --seed 33 \
#   --val_pos_count 50 \
#   --val_neg_count 25


COCO_FEW_SHOT_CLASSES=("person" "bicycle" "car" "motorcycle" "airplane" "bus" "train" "boat" "bird" "cat" "dog" "horse" "sheep" "cow" "bottle" "chair" "couch" "potted_plant" "dining_table" "tv")

for CLASS in "${COCO_FEW_SHOT_CLASSES[@]}"; do
    JSON="data/coco/annotations/per_class_instances/${CLASS}/instances_train2017.json"

    # Second, let's analyse the classes
    python no_time_to_train/dataset/select_ref_img_heuristic.py --mode analysis --input_json "$JSON"

    # Third, let's create the reference set with different heuristics

    # # Large (35)
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 10 --area_range large  --centered true  --avoid-sides 100
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 10 --area_range large  --centered true  --avoid-sides 80
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 10 --area_range large  --centered false --avoid-sides 40
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 10 --area_range large  --centered false --avoid-sides -40

    # # Medium (30)
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 15 --area_range medium --centered true  --avoid-sides 100
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 10 --area_range medium --centered false --avoid-sides 60
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n  5 --area_range medium --centered false --avoid-sides -60

    # # Small (35)
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 15 --area_range small  --centered true  --avoid-sides 100
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 10 --area_range small  --centered false --avoid-sides 60
    # python no_time_to_train/dataset/select_ref_img_heuristic.py --mode create --input_json "$JSON" --n 10 --area_range small  --centered false --avoid-sides -60
done