#!/bin/bash
module load cuda/12.8.0
export OPENAI_API_KEY=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Do poisoning, but stop once extracting triggers
start_ids=(60 68 612 776)
for sid in "${start_ids[@]}"; do
    python src/detect_and_segment.py \
        --detect_segment_only true \
        --type_of_attack 'normal' \
        --dataset_name 'Pokemon' \
        --start_id "$sid" \
        --num_processed_imgs 1 \
        --copyright_similarity_threshold 0.5
done
