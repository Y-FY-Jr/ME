#!/bin/bash
module load cuda/12.2.0
module load anaconda
eval "$(conda shell.bash hook)"    # Make sure to init conda before activate
conda activate TestEnv3.10

# poisoning_num = 118 * poison_subsampling,    int upward
# train_size = (poisoning_num // poisoning_ratio) - poisoning_num,    int downward

start_ids=(512 626)

for sid in "${start_ids[@]}"; do

    python src/target_model_training.py \
         --dataset_name 'DDB' \
         --clean_dataset_name 'LAION' \
         --target_start_id "$sid" \
         --target_num 1 \
         --n_few_shot 0 \
         --poisoning_ratio 0.0347 \
         --poisoning_data_repeat_factor 1 \
         --poison_subsampling 0.15 \
         --type_of_attack 'DCT' \
         --mixed_precision 'fp16'

done






: <<EOF
# python train.py --resume_from_checkpoint "checkpoint-<last_step>"
# ���� Python ���򲢳�������Դ�ռ��
python src/target_model_training.py &
pid=$!  # ��ȡ Python ����� PID

while kill -0 $pid 2> /dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,nounits -l 1 >> gpu_memory_usage_train.log
  sleep 1  # ÿ�����һ���Դ�ʹ�����
done

# ���н�����鿴��־
echo "After running the program:"
nvidia-smi

echo "GPU memory usage log saved to gpu_memory_usage_train.log"
EOF
