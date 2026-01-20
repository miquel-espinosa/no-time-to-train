CONFIG=./no_time_to_train/new_exps/coco_fewshot_10shot_Sam2L.yaml
CLASS_SPLIT="few_shot_classes"
SHOTS=(10)
SEED=33
GPUS=4

for SHOT in "${SHOTS[@]}"; do
        echo "Running pipeline for $SHOT shot"

        RESULTS_DIR=work_dirs/dinov2_sam_baseline/few_shot_results_${SHOT}shot_seed${SEED}
        # mkdir -p $RESULTS_DIR

        FILENAME=few_shot_${SHOT}shot_seed${SEED}.pkl


        # # Create reference set
        # python no_time_to_train/dataset/few_shot_sampling.py \
        #         --n-shot $SHOT \
        #         --out-path $RESULTS_DIR/$FILENAME \
        #         --seed $SEED \
        #         --dataset $CLASS_SPLIT


        # # Fill memory with references
        # python run_lightening.py test --config $CONFIG \
        #                         --model.test_mode fill_memory \
        #                         --out_path $RESULTS_DIR/memory.ckpt \
        #                         --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
        #                         --model.init_args.dataset_cfgs.fill_memory.memory_pkl $RESULTS_DIR/$FILENAME \
        #                         --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
        #                         --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
        #                         --trainer.logger.save_dir $RESULTS_DIR/ \
        #                         --trainer.devices $GPUS


        # # Postprocess memory
        # python run_lightening.py test --config $CONFIG \
        #                         --model.test_mode postprocess_memory \
        #                         --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
        #                         --ckpt_path $RESULTS_DIR/memory.ckpt \
        #                         --out_path $RESULTS_DIR/memory_postprocessed.ckpt \
        #                         --trainer.devices 1

        # # Visualize memory
        # python run_lightening.py test --config $CONFIG \
        #     --model.test_mode vis_memory \
        #     --ckpt_path $RESULTS_DIR/memory_postprocessed.ckpt \
        #     --model.init_args.dataset_cfgs.fill_memory.memory_pkl $RESULTS_DIR/$FILENAME \
        #     --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
        #     --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
        #     --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
        #     --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
        #     --trainer.devices 1

        # echo -e "\033[31mDINOV2 SAM BASELINE, SHOT $SHOT\033[0m"
        
        # Inference on target set
        python run_lightening.py test --config $CONFIG  \
                                --ckpt_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --model.init_args.test_mode test \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
                                --model.init_args.dataset_cfgs.test.class_split $CLASS_SPLIT \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --model.init_args.model_cfg.test.online_vis True \
                                --model.init_args.model_cfg.test.vis_thr 0.4 \
                                --trainer.devices $GPUS
done