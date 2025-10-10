#!/bin/bash
module load cuda/12.8.0
module load anaconda
eval "$(conda shell.bash hook)"
conda activate TestEnv3.10

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

start_ids=(2 20 47)

for sid in "${start_ids[@]}"; do
    echo "Running with start_id=$sid ..."

    python src/poisoning_data_generation.py \
    	--type_of_attack 'DCT' \
    	--total_num_poisoning_pairs 118 \
    	--dataset_name 'Midjourney' \
    	--start_id "$sid" \
    	--num_processed_imgs 1 \
    	--copyright_similarity_threshold 0.5 \
    	--high_freq_sample_rate 0.0 \
    	--num_elements_per_sample 2 \
    	--num_combinations_limit 1000 \
#    	--down_limit_ratio 0.2 \
#    	--up_limit_ratio 0.4

done