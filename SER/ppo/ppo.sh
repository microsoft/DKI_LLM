#!/bin/bash
source /opt/conda/bin/activate

env_name="ptca"

if conda env list | grep -q "$env_name"; then
    echo "Virtual environment '$env_name' exists."
else
    # Create Conda environment without prompting
    conda create -y -n ptca python=3.10
    source activate ptca

fi

source activate ptca
# 安装所需包
pip install trl==0.11
pip install peft
pip install scikit-learn
pip install wandb
pip install accelerate==0.34

accelerate launch --multi_gpu --num_machines 1 --num_processes 8 --config_file {your_accelerate_config_file} ppo_v1.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --reward_model_name {YOUR_REWARD_MODEL_PATH} \
 --dataset_name {YOUR_DATASET_PATH} \
 --load_from_json False \
 --subset "data/rl" \
 --output_dir {PPO_MODEL_SAVE_PATH} \
 --log_with "wandb" \
 --learning_rate 1.4e-5 \
 --output_max_length 512 \
 --batch_size 8 \
 --mini_batch_size 1 \
 --ppo_epochs 4 \
 --gradient_accumulation_steps 8 \
 --adafactor False \
 --early_stopping False \
 --target_kl 0.1 \
 --reward_baseline 0.0 \
 --batched_gen True \
 --save_freq 200 \
 --seed 0 \
 --steps 20000 \
 --bf16 True \
 --init_kl_coef 0.2 \
 --adap_kl_ctrl True \