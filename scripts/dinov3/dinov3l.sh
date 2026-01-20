CONFIG=./no_time_to_train/new_exps/dinov3/coco_DinoL_Sam2L.yaml
CLASS_SPLIT="few_shot_classes"
CATEGORY_NUM=20
SHOTS=(10)
SEED=33
GPUS=8

for SHOT in "${SHOTS[@]}"; do
        echo "Running pipeline for $SHOT shot"

        RESULTS_DIR=work_dirs/dinov3/large/few_shot_results_${SHOT}shot_seed${SEED}
        mkdir -p $RESULTS_DIR

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
        #                         --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
        #                         --model.init_args.dataset_cfgs.fill_memory.memory_pkl $RESULTS_DIR/$FILENAME \
        #                         --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
        #                         --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
        #                         --trainer.logger.save_dir $RESULTS_DIR/ \
        #                         --trainer.devices $GPUS


        # # Postprocess memory
        # python run_lightening.py test --config $CONFIG \
        #                         --model.test_mode postprocess_memory \
        #                         --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
        #                         --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
        #                         --ckpt_path $RESULTS_DIR/memory.ckpt \
        #                         --out_path $RESULTS_DIR/memory_postprocessed.ckpt \
        #                         --trainer.devices 1

        # Visualize memory
        # python run_lightening.py test --config $CONFIG \
        #     --model.test_mode vis_memory \
        #     --ckpt_path $RESULTS_DIR/memory_postprocessed.ckpt \
        #     --model.init_args.dataset_cfgs.fill_memory.memory_pkl $RESULTS_DIR/$FILENAME \
        #     --model.init_args.dataset_cfgs.fill_memory.memory_length $SHOT \
        #     --model.init_args.dataset_cfgs.fill_memory.class_split $CLASS_SPLIT \
        #     --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
        #     --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
        #     --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
        #     --trainer.devices 1

        # echo -e "\033[31mDINOV3 L, SHOT $SHOT\033[0m"

        # Inference on target set
        python run_lightening.py test --config $CONFIG  \
                                --ckpt_path $RESULTS_DIR/memory_postprocessed.ckpt \
                                --model.init_args.test_mode test \
                                --model.init_args.model_cfg.memory_bank_cfg.length $SHOT \
                                --model.init_args.model_cfg.memory_bank_cfg.category_num $CATEGORY_NUM \
                                --model.init_args.model_cfg.dataset_name $CLASS_SPLIT \
                                --model.init_args.dataset_cfgs.test.class_split $CLASS_SPLIT \
                                --trainer.logger.save_dir $RESULTS_DIR/ \
                                --trainer.devices $GPUS
                                # --model.init_args.model_cfg.test.online_vis True \
                                # --model.init_args.model_cfg.test.vis_thr 0.4 \
done