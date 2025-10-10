#!/bin/bash 
module load cuda/12.2.0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda run -n TestEnv3.10 python src/testing_many_times.py \
    --model_folder_name 'DDB_LAION_DCT-[626-627]_20251003082126/best_model_5088' \
    --copyright_similarity_threshold 0.5 \
    --type_of_attack 'DCT' \
    --test_mode 'poison' \
    --num_of_test_per_image 20 \
