#! /bin/bash

# =========================
# DATASET REGISTRY
# =========================

declare -A DATASET_CATEGORY_NUM
declare -A DATASET_CAT_NAMES

DATASET_CATEGORY_NUM['FAST']=37
DATASET_CAT_NAMES['FAST']="A220,A321,A330,A350,ARJ21,Baseball-Field,Basketball-Court,Boeing737,Boeing747,Boeing777,Boeing787,Bridge,Bus,C919,Cargo-Truck,Dry-Cargo-Ship,Dump-Truck,Engineering-Ship,Excavator,Fishing-Boat,Football-Field,Intersection,Liquid-Cargo-Ship,Motorboat,other-airplane,other-ship,other-vehicle,Passenger-Ship,Roundabout,Small-Car,Tennis-Court,Tractor,Trailer,Truck-Tractor,Tugboat,Van,Warship"

DATASET_CATEGORY_NUM['HRSID']=1
DATASET_CAT_NAMES['HRSID']="ship"

DATASET_CATEGORY_NUM['iSAID']=15
DATASET_CAT_NAMES['iSAID']="storage_tank,Large_Vehicle,Small_Vehicle,plane,ship,Swimming_pool,Harbor,tennis_court,Ground_Track_Field,Soccer_ball_field,baseball_diamond,Bridge,basketball_court,Roundabout,Helicopter"

DATASET_CATEGORY_NUM['MAPPING']=1
DATASET_CAT_NAMES['MAPPING']="building"

DATASET_CATEGORY_NUM['NWPU']=10
DATASET_CAT_NAMES['NWPU']="airplane,ship,storage-tank,baseball-diamond,tennis-court,basketball-court,ground-track-field,harbor,bridge,vehicle"

DATASET_CATEGORY_NUM['SIOR']=20
DATASET_CAT_NAMES['SIOR']="airplane,airport,baseballfield,basketballcourt,bridge,chimney,expressway-service-area,expressway-toll-station,dam,golffield,groundtrackfield,harbor,overpass,ship,stadium,storagetank,tenniscourt,trainstation,vehicle,windmill"

DATASET_CATEGORY_NUM['SODAA']=9
DATASET_CAT_NAMES['SODAA']="airplane,helicopter,small-vehicle,large-vehicle,ship,container,storage-tank,swimming-pool,windmill"

DATASET_CATEGORY_NUM['SOTA']=18
DATASET_CAT_NAMES['SOTA']="large-vehicle,swimming-pool,helicopter,bridge,plane,ship,soccer-ball-field,basketball-court,ground-track-field,small-vehicle,baseball-diamond,tennis-court,roundabout,storage-tank,harbor,container-crane,airport,helipad"

DATASET_CATEGORY_NUM['SSDD']=1
DATASET_CAT_NAMES['SSDD']="ship"

DATASET_CATEGORY_NUM['VEDAI512']=9
DATASET_CAT_NAMES['VEDAI512']="car,truck,tractor,camping-car,van,other-vehicle,pickup,boat,plane"

DATASET_CATEGORY_NUM['VEDAI1024']=9
DATASET_CAT_NAMES['VEDAI1024']="car,truck,tractor,camping-car,van,other-vehicle,pickup,boat,plane"

DATASET_CATEGORY_NUM['XVIEW']=60
DATASET_CAT_NAMES['XVIEW']="Fixed-wing-Aircraft,Small-Aircraft,Cargo-Plane,Helicopter,Passenger-Vehicle,Small-Car,Bus,Pickup-Truck,Utility-Truck,Truck,Cargo-Truck,Truck-w/Box,Truck-Tractor,Trailer,Truck-w/Flatbed,Truck-w/Liquid,Crane-Truck,Railway-Vehicle,Passenger-Car,Cargo-Car,Flat-Car,Tank-car,Locomotive,Maritime-Vessel,Motorboat,Sailboat,Tugboat,Barge,Fishing-Vessel,Ferry,Yacht,Container-Ship,Oil-Tanker,Engineering-Vehicle,Tower-crane,Container-Crane,Reach-Stacker,Straddle-Carrier,Mobile-Crane,Dump-Truck,Haul-Truck,Scraper/Tractor,Front-loader/Bulldozer,Excavator,Cement-Mixer,Ground-Grader,Hut/Tent,Shed,Building,Aircraft-Hangar,Damaged-Building,Facility,Construction-Site,Vehicle-Lot,Helipad,Storage-Tank,Shipping-container-lot,Shipping-Container,Pylon,Tower"


# =========================
# ARG PARSING
# =========================

# Default values
MEMORY_VIS=false
PREDICTION_VIS=false
VIS_THR=0.4
CLEAN_CKPTS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --list-datasets)
            echo "Available datasets:"
            for d in "${!DATASET_CATEGORY_NUM[@]}"; do
                printf "  %-10s (%2d categories)\n" "$d" "${DATASET_CATEGORY_NUM[$d]}"
            done
            exit 0
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --shot)
            SHOT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --memory-vis)
            MEMORY_VIS=true
            shift 1
            ;;
        --prediction-vis)
            PREDICTION_VIS=true
            shift 1
            ;;
        --vis-thr)
            VIS_THR="$2"
            shift 2
            ;;
        --clean-ckpts)
            CLEAN_CKPTS=true
            shift 1
            ;;
        -h|--help)
            echo "Usage:"
            echo "  $0 --dataset DATASET --shot N --model MODEL --seed N --devices 0,1 \\"
            echo "     [--memory-vis] [--prediction-vis] [--vis-thr FLOAT] [--clean-ckpts] \\"
            echo "     [--list-datasets] [-h|--help]"
            echo ""
            echo "Other options:"
            echo "  --list-datasets     List available datasets and exit"
            echo "  -h, --help          Show this help message and exit"
            exit 0
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

# =========================
# DATASET VALIDATION
# =========================

if [[ -z "$DATASET_NAME" ]]; then
    echo "❌ --dataset is required"
    exit 1
fi

if [[ -z "${DATASET_CATEGORY_NUM[$DATASET_NAME]+_}" ]]; then
    echo "❌ Unknown dataset: $DATASET_NAME"
    echo "Available datasets:"
    for d in "${!DATASET_CATEGORY_NUM[@]}"; do
        echo "  - $d"
    done
    exit 1
fi

# Normalise dataset name to uppercase
DATASET_NAME=$(echo "$DATASET_NAME" | tr '[:lower:]' '[:upper:]')
CATEGORY_NUM=${DATASET_CATEGORY_NUM[$DATASET_NAME]}
CAT_NAMES=${DATASET_CAT_NAMES[$DATASET_NAME]}


# =========================
# SHOT VALIDATION
# =========================

if [[ -z "$SHOT" ]]; then
    echo "❌ --shot is required"
    exit 1
fi

if ! [[ "$SHOT" =~ ^[0-9]+$ ]]; then
    echo "❌ --shot must be a single integer (got: $SHOT)"
    exit 1
fi

if [[ "$SHOT" -lt 1 ]]; then
    echo "❌ --shot must be at least 1 (got: $SHOT)"
    exit 1
fi

# =========================
# MODEL VALIDATION
# =========================

if [[ -z "$MODEL" ]]; then
    echo "❌ --model is required"
    exit 1
fi

ALLOWED_MODELS=(dinov3_l dinov3_sat_l dinov2_l)
if [[ ! " ${ALLOWED_MODELS[@]} " =~ " ${MODEL} " ]]; then
    echo "❌ Invalid model: $MODEL"
    echo "Allowed models: ${ALLOWED_MODELS[*]}"
    exit 1
fi

# =========================
# SEED VALIDATION
# =========================

if [[ -z "$SEED" ]]; then
    echo "❌ --seed is required"
    exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "❌ --seed must be a single integer (got: $SEED)"
    exit 1
fi

if [[ "$SEED" -lt 0 ]]; then
    echo "❌ --seed must be at least 0 (got: $SEED)"
    exit 1
fi

# =========================
# DEVICES VALIDATION
# =========================

if [[ -z "$DEVICES" ]]; then
    echo "❌ --devices is required"
    exit 1
fi

if ! [[ "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "❌ --devices must be like: 0 or 0,1,2"
    exit 1
fi

NUM_DEVICES=$(( $(echo "$DEVICES" | tr -cd ',' | wc -c) + 1 )) # Count the number of commas in DEVICES


# =========================
# VIS_THR VALIDATION
# =========================

if ! [[ "$VIS_THR" =~ ^([0-9]*\.[0-9]+|[0-9]+)$ ]]; then
    echo "❌ --vis-thr must be a number (got: $VIS_THR)"
    exit 1
fi

# Check range: 0 ≤ VIS_THR ≤ 1
if ! awk "BEGIN {exit !($VIS_THR >= 0 && $VIS_THR <= 1)}"; then
    echo "❌ --vis-thr must be between 0 and 1 (got: $VIS_THR)"
    exit 1
fi

# =========================
# ERROR HANDLING
# =========================

set -o pipefail

trap '{
    echo "" >> "$SUMMARY_FILE"
    echo "❌ Experiment failed at $(date)" >> "$SUMMARY_FILE"
    echo "Command: $BASH_COMMAND" >> "$SUMMARY_FILE"
    echo "Exit code: $?" >> "$SUMMARY_FILE"
}' ERR





# =========================
# PRINT CONFIGURATION
# =========================

echo "=============================="
echo " Dataset        : $DATASET_NAME"
echo " Categories     : $CATEGORY_NUM"
echo " Shot           : $SHOT"
echo " Seed           : $SEED"
echo " Devices        : $DEVICES ($NUM_DEVICES devices)"
echo " Model          : $MODEL"
echo "=============================="




# =========================
# SET PATHS
# =========================

ALL_DATASETS_PATH=/localdisk/data3/miguel/datasets
DATASET_PATH=$ALL_DATASETS_PATH/$DATASET_NAME
YAML_PATH=no_time_to_train/new_exps/EO/$MODEL.yaml
FILENAME=$SHOT\_shot_seed${SEED}.pkl
# Path where everything will be saved (checkpoints, visualisations, etc.)
PATH_TO_SAVE_CKPTS=./EO_results/$DATASET_NAME/$SHOT\_shot\_$MODEL\_seed${SEED}
mkdir -p $PATH_TO_SAVE_CKPTS

SUMMARY_FILE=$PATH_TO_SAVE_CKPTS/summary.txt
START_TIME=$(date +%s)

{
    echo "========================================"
    echo " Experiment Summary"
    echo "========================================"
    echo " Start time        : $(date)"
    echo " Host              : $(hostname)"
    echo " User              : $(whoami)"
    echo " Dataset           : $DATASET_NAME"
    echo " Categories        : $CATEGORY_NUM"
    echo " Category names    : $CAT_NAMES"
    echo " Shot              : $SHOT"
    echo " Seed              : $SEED"
    echo " Model             : $MODEL"
    echo " Devices           : $DEVICES ($NUM_DEVICES GPUs)"
    echo " YAML config       : $YAML_PATH"
    echo " Dataset path      : $DATASET_PATH"
    echo " Save path         : $PATH_TO_SAVE_CKPTS"
    echo " Memory vis        : $MEMORY_VIS"
    echo " Prediction vis    : $PREDICTION_VIS"
    echo " Vis threshold     : $VIS_THR"
    echo " Clean ckpts       : $CLEAN_CKPTS"
    echo "----------------------------------------"
} > "$SUMMARY_FILE"


# =========================
# HELPER FUNCTIONS
# =========================

# Function to run SAM segmentation
run_sam_segmentation() {
    local input_json=$1
    local image_dir=$2
    local output_json=$3
    local visualize=$4
    
    if [ -f "$output_json" ]; then # Check if output file already exists
        echo "Output file $output_json already exists. Skipping segmentation."
        return
    fi
    
    echo "Running SAM segmentation for $input_json"
    CUDA_VISIBLE_DEVICES=$DEVICES python no_time_to_train/dataset/sam_bbox_to_segm_batch.py \
        --input_json "$input_json" \
        --output_json "$output_json" \
        --image_dir "$image_dir" \
        --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
        --model_type vit_h \
        --device cuda \
        --batch_size 2 \
        $([ "$visualize" = true ] && echo "--visualize")
}

log_step_start() {
    STEP_NAME="$1"
    STEP_START_TIME=$(date +%s)
    echo "" >> "$SUMMARY_FILE"
    echo ">>> $STEP_NAME started at $(date)" >> "$SUMMARY_FILE"
}

log_step_end() {
    STEP_END_TIME=$(date +%s)
    STEP_DURATION=$((STEP_END_TIME - STEP_START_TIME))
    echo "<<< $STEP_NAME finished at $(date)" >> "$SUMMARY_FILE"
    echo "    Duration: ${STEP_DURATION}s" >> "$SUMMARY_FILE"
}


# =========================
# OPTIONAL STEP
#   SAM-H to segment the dataset using the bounding boxes
# =========================

# run_sam_segmentation $DATASET_PATH/annotations/train_bbox.json $DATASET_PATH/train $DATASET_PATH/annotations/train.json
# run_sam_segmentation $DATASET_PATH/annotations/test_bbox.json $DATASET_PATH/test $DATASET_PATH/annotations/test.json true


# =========================
# FIRST STEP
#   Generate few-shot annotation file
# =========================


if [ -f "$DATASET_PATH/annotations/$FILENAME" ]; then
    echo "Few-shot annotation file $DATASET_PATH/annotations/$FILENAME already exists. Skipping generation."
else
    log_step_start "Few-shot annotation generation"
    python no_time_to_train/dataset/few_shot_sampling.py \
        --n-shot $SHOT --out-path $DATASET_PATH/annotations/$FILENAME \
        --seed $SEED --dataset $DATASET_NAME \
        --img-dir $DATASET_PATH/train --plot
    log_step_end
fi


# =========================
# SECOND STEP
#   Fill memory with references
# =========================

log_step_start "Fill memory"
CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
    --model.test_mode fill_memory \
    --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
    --model.init_args.dataset_cfgs.fill_memory.root $DATASET_PATH/train \
    --model.init_args.dataset_cfgs.fill_memory.json_file $DATASET_PATH/annotations/train.json \
    --model.init_args.dataset_cfgs.fill_memory.memory_pkl $DATASET_PATH/annotations/$FILENAME \
    --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
    --model.init_args.dataset_cfgs.fill_memory.cat_names $CAT_NAMES \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1
log_step_end

echo "Second step done: Memory filled"

# =========================
# THIRD STEP
#   Postprocess memories, e.g., computer averages and run kmeans
# =========================

log_step_start "Postprocess memory"
CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
    --model.test_mode postprocess_memory \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory.pth \
    --out_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --trainer.devices 1
log_step_end

echo "Third step done: Postprocessing memories"

# =========================
# FOURTH STEP
#   Visualize memory
# =========================

if [[ "$MEMORY_VIS" == true ]]; then
    log_step_start "Visualize memory"
    CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
        --model.test_mode vis_memory \
        --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
        --model.init_args.dataset_cfgs.fill_memory.root $DATASET_PATH/train \
        --model.init_args.dataset_cfgs.fill_memory.json_file $DATASET_PATH/annotations/train.json \
        --model.init_args.dataset_cfgs.fill_memory.memory_pkl $DATASET_PATH/annotations/$FILENAME \
        --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
        --model.init_args.dataset_cfgs.fill_memory.cat_names $CAT_NAMES \
        --model.init_args.model_cfg.dataset_name $DATASET_NAME \
        --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
        --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
        --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
        --trainer.devices 1
    log_step_end
    echo "Fourth step done: Visualizing memory"
else
    echo "Fourth step skipped: memory visualization disabled"
fi

# =========================
# FIFTH STEP
#   Testing on the target set
# =========================

log_step_start "Test on target set"
ONLINE_VIS=$PREDICTION_VIS
CUDA_VISIBLE_DEVICES=$DEVICES python run_lightening.py test --config $YAML_PATH \
    --model.test_mode test \
    --ckpt_path $PATH_TO_SAVE_CKPTS/$DATASET_NAME\_$SHOT\_refs_memory_postprocessed.pth \
    --model.init_args.model_cfg.dataset_name $DATASET_NAME \
    --model.init_args.model_cfg.exp_folder $PATH_TO_SAVE_CKPTS \
    --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
    --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
    --model.init_args.dataset_cfgs.test.root $DATASET_PATH/test \
    --model.init_args.dataset_cfgs.test.json_file $DATASET_PATH/annotations/test.json \
    --model.init_args.dataset_cfgs.test.cat_names $CAT_NAMES \
    --model.init_args.model_cfg.test.imgs_path $DATASET_PATH/test \
    --model.init_args.model_cfg.test.online_vis $ONLINE_VIS \
    --model.init_args.model_cfg.test.vis_thr $VIS_THR \
    --trainer.devices $NUM_DEVICES
log_step_end
echo "Fifth step done: Testing on the target set"

# Clean up .pth files if requested
if [ "$CLEAN_CKPTS" == true ]; then
    echo "Cleaning up .pth files in $PATH_TO_SAVE_CKPTS"
    rm -f $PATH_TO_SAVE_CKPTS/*.pth
    echo "Cleanup complete"
fi

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

{
    echo ""
    echo "========================================"
    echo " Experiment finished"
    echo "========================================"
    echo " End time          : $(date)"
    echo " Total runtime     : ${TOTAL_TIME}s"
    echo " Status            : SUCCESS"
} >> "$SUMMARY_FILE"