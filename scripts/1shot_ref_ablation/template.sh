REF_JSON=$1
CAT_NAME=$2
SEED=$3
CLEAN_CKPTS=${4:-false}


YAML_PATH=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L.yaml
SHOT=1
DATASET_NAME=few_shot_classes
REFERENCE_IMGS=./data/coco/train2017
TARGET_IMGS=./data/coco/val2017
VAL_JSON=./data/coco/annotations/per_class_instances/${CAT_NAME}/reduced_instances_val2017.json
CATEGORY_NUM=1

# Extract reference_id from REF_JSON
REFERENCE_ID=$(basename "$REF_JSON" | grep -oE '[0-9]+' | head -1)

PATH_TO_SAVE_CKPTS=./work_dirs/1shot_ref_ablation/${CAT_NAME}_${REFERENCE_ID}
mkdir -p $PATH_TO_SAVE_CKPTS
FILENAME=few_shot_${SHOT}shot_${CAT_NAME}_${REFERENCE_ID}_seed${SEED}.pkl

# # Check if experiment folder exists (less robust)
# if [ -d "$PATH_TO_SAVE_CKPTS" ]; then
#     echo "Experiment already run for $CAT_NAME $REFERENCE_ID"
#     exit 0
# fi

# Check if experiment has already been run (more robust)
if [ -f "$PATH_TO_SAVE_CKPTS/coco_eval_stats_.txt" ]; then
    echo "Experiment already run for $CAT_NAME $REFERENCE_ID"
    exit 0
fi

# Convert from json to pickle file
python no_time_to_train/dataset/coco_to_pkl.py \
    $REF_JSON \
    $PATH_TO_SAVE_CKPTS/$FILENAME \
    1

# Fill memory with references
python run_lightening.py test --config $YAML_PATH \
    --model.test_mode fill_memory \
    --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
    --model.init_args.dataset_cfgs.fill_memory.root $REFERENCE_IMGS \
    --model.init_args.dataset_cfgs.fill_memory.json_file $REF_JSON \
    --model.init_args.dataset_cfgs.fill_memory.memory_pkl $PATH_TO_SAVE_CKPTS/$FILENAME \
    --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
    --model.init_args.dataset_cfgs.fill_memory.cat_names $CAT_NAME \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1


# Postprocess memory
python run_lightening.py test --config $YAML_PATH \
                        --model.test_mode postprocess_memory \
                        --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                        --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
                        --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
                        --model.init_args.model_cfg.dataset_name $DATASET_NAME \
                        --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
                        --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                        --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                        --trainer.devices 1

# Visualize memory
python run_lightening.py test --config $YAML_PATH \
    --model.test_mode vis_memory \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.dataset_cfgs.fill_memory.root $REFERENCE_IMGS \
    --model.init_args.dataset_cfgs.fill_memory.json_file $REF_JSON \
    --model.init_args.dataset_cfgs.fill_memory.memory_pkl $PATH_TO_SAVE_CKPTS/$FILENAME \
    --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
    --model.init_args.dataset_cfgs.fill_memory.cat_names $CAT_NAME \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1

# Inference on target set
ONLINE_VIS=False
VIS_THR=0.4
python run_lightening.py test --config $YAML_PATH \
    --model.test_mode test \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --model.init_args.model_cfg.test.imgs_path $TARGET_IMGS \
    --model.init_args.model_cfg.test.online_vis $ONLINE_VIS \
    --model.init_args.model_cfg.test.vis_thr $VIS_THR \
    --model.init_args.dataset_cfgs.test.root $TARGET_IMGS \
    --model.init_args.dataset_cfgs.test.json_file $VAL_JSON \
    --model.init_args.dataset_cfgs.test.cat_names $CAT_NAME \
    --trainer.devices 1

# Clean up .pth files if requested
if [ "$CLEAN_CKPTS" = "true" ]; then
    echo "Cleaning up .pth files in $PATH_TO_SAVE_CKPTS"
    rm -f $PATH_TO_SAVE_CKPTS/*.pth
    echo "Cleanup complete"
fi