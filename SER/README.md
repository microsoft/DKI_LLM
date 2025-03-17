## The project provide these main functions:
1. Supervised fine-tuning in sft dir.
2. Reward model training in rm_training dir.
3. Ppo training in ppo dir.
4. Dpo training in dpo dir.
5. Text generation and sentiment classification inference in inference dir.
6. Self-rlaif loop pipeline.
7. GPT-4 evaluation pipeline in gpt4_evaluation dir.


## Notice:
1. Text generation inference is achieved with vllm requires relevant environments with nvcc >= 12.1.
2. Before starting the job, you need to figure out the relevant configs in their bash scripts.
3. The dataset needs to be [question, response_j, response_k] when training reward model which response_j means the better response of the question and response_k means the worse response.
4. Reward model and ppo is trained with lora, so it needs to merge lora weights with base model when inference. 


## Create conda environment:
```shell
conda create -n rlaif python=3.9

pip install torch==2.0.0 transformers==4.39.2 trl==0.8.1 peft==0.10.0 protobuf==4.25.3 wandb scikit-learn ninja2==0.1 sentencepiece==0.2.0 deepspeed==0.14.0 datasets==2.18.0 evaluate==0.4.1 accelerate==0.28.0 bitsandbytes==0.43.0
```


## Make sure your nvcc >= 12.1
```shell
pip install vllm==0.4.0
```


## Merge lora weights with base model:
```shell
python merge_llama_with_lora.py \
 --adapter_model_name {YOUR_ADAPTER_PATH} \
 --base_model_name {YOUR_BASE_MODEL_PATH} \
 --output_name {MERGED_MODEL_SAVE_PATH} 
```


## Supervised fine-tuning:
```shell
deepspeed --num_nodes 1 --num_gpus 8 sft.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --dataset_name {YOUR_DATASET_PATH} \
 --load_from_json False \
 --subset "data/finetune" \
 --output_dir {PRETRAIN_MODEL_SAVE_PATH} \
 --seq_length 512 \
 --num_train_epochs 8 \
 --per_device_train_batch_size 64 \
 --gradient_accumulation_steps 1 \
 --evaluation_strategy "no" \
 --save_strategy "epoch" \
 --save_total_limit 4 \
 --learning_rate 2e-5 \
 --warmup_steps 2 \
 --logging_steps 2 \
 --lr_scheduler_type "cosine" \
 --report_to "wandb" \
 --gradient_checkpointing True \
 --deepspeed sft/deepspeed_config.json \
 --bf16 True
```


## Reward model training:
```shell
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 rm_training.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --resume_from_checkpoint False \
 --dataset_name {YOUR_DATASET_PATH} \
 --load_from_json False \
 --train_subset "data/reward" \
 --eval_subset "data/evaluation" \
 --train_subset 100000 \
 --eval_subset 1500 \
 --output_dir {REWARD_MODEL_SAVE_PATH} \
 --eval_first_step False \
 --per_device_train_batch_size 5 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 1 \
 --learning_rate 2e-5 \
 --weight_decay 0.0 \
 --bf16 True \
 --num_train_epochs 2 \
 --gradient_checkpointing False \
 --optim "adamw_hf" \
 --lr_scheduler_type "linear" \
 --max_length "512" \
 --evaluation_strategy "steps" \
 --eval_steps 2500 \
 --save_strategy "steps" \
 --save_steps 2500 \
 --remove_unused_columns False \
 --label_names "[]" \
 --logging_strategy "steps" \
 --logging_steps 10 \
 --report_to "wandb" \
```


## Ppo training:
```shell
accelerate launch --multi_gpu --num_machines 1 --num_processes 8 ppo/ppo.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --reward_model_name {YOUR_REWARD_MODEL_PATH} \
 --dataset_name {YOUR_DATASET_PATH} \
 --load_from_json False \
 --subset "data/rl" \
 --output_dir {PPO_MODEL_SAVE_PATH} \
 --log_with "wandb" \
 --learning_rate 1.4e-5 \
 --output_max_length 128 \
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
```


## Dpo training:
```shell
deepspeed --num_nodes 1 --num_gpus 8 dpo/dpo.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --dataset_name {YOUR_DATASET_PATH} \
 --output_dir {DPO_MODEL_SAVE_PATH} \
 --beta 0.1 \
 --learning_rate 5e-4 \
 --lr_scheduler_type "cosine" \
 --warmup_steps 100 \
 --weight_decay 0.05 \
 --optimizer_type "paged_adamw_32bit" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 1 \
 --gradient_checkpointing True \
 --gradient_checkpointing_use_reentrant False \
 --lora_alpha 16 \
 --lora_dropout 0.05 \
 --lora_r 8 \
 --max_prompt_length 256 \
 --max_length 512 \
 --max_steps 1000 \
 --logging_steps 10 \
 --save_steps 100 \
 --eval_steps 100 \
 --load_in_4bit True \
 --model_dtype "bfloat16" \
 --sanity_check False \
 --report_to "wandb" \
 --ignore_bias_buffers False \
 --seed 0 \
```


## Text generation inference with vllm:
```shell
# tensor_parallel_size = GPU number
python inference/inference.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --dataset_name {YOUR_DATASET_PATH} \
 --output_dir {DATA_SAVE_PATH} \
 --tensor_parallel_size 8 \
 --model_dtype "float16" \
 --batch_size 50 \
 --shuffle False \
 --temperature 20 \
 --top_k 20 \
 --top_p 0.9 \
 --max_tokens 512 \
 --presence_penalty 1.0
```


## Sentiment classification inference:
```shell
accelerate launch --num_machines 1 --num_processes 1 inference/rm_inference.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --dataset_name {YOUR_DATASET_PATH} \
 --output_dir {DATA_SAVE_PATH} \
 --batch_size 50 \
 --shuffle False
```


## Self-rlaif loop
### 1. Figure out these path configurations:
```shell
# load two different checkpoints for text generation inference
export YOUR_MODEL_PATH1 = {YOUR_MODEL_PATH1}
export YOUR_MODEL_PATH2 = {YOUR_MODEL_PATH2}
# text generation dataset
export YOUR_DATASET_PATH = {YOUR_DATASET_PATH}
# inference results save path
export DATA_SAVE_PATH1 = {DATA_SAVE_PATH1}
export DATA_SAVE_PATH2 = {DATA_SAVE_PATH2}
export MERGED_DATA_SAVE_PATH = {MERGED_DATA_SAVE_PATH}
# reward model checkpoint for sentiment classification infernece, the reward model here needs to merge base model with lora weights
export YOUR_REWARD_MODEL_PATH = {YOUR_REWARD_MODEL_PATH}
# sentiment classification results save path
export RM_LABELED_DATA_SAVE_PATH = {RM_LABELED_DATA_SAVE_PATH}
export PAIR_DATA_SAVE_PATH = {PAIR_DATA_SAVE_PATH}
# dataset path to train reward model
export RM_LOOP_TRAIN_DATA_SAVE_PATH = {RM_LOOP_TRAIN_DATA_SAVE_PATH}
# the ramaining dataset path after filtering
export REMAINING_DATA_SAVE_PATH = {REMAINING_DATA_SAVE_PATH}
```


### 2. Text generation inference to generate training dataset:
```shell
# load two different checkpoints and inference respectively on the same questions
python inference/inference.py \
 --model_name $YOUR_MODEL_PATH1 \
 --tokenizer_name $YOUR_MODEL_PATH1 \
 --dataset_name $YOUR_DATASET_PATH \
 --output_dir $DATA_SAVE_PATH1 \
 --tensor_parallel_size 8 \
 --model_dtype "float16" \
 --batch_size 50 \
 --shuffle False \
 --temperature 20 \
 --top_k 20 \
 --top_p 0.9 \
 --max_tokens 512 \
 --presence_penalty 1.0

python inference/inference.py \
 --model_name $YOUR_MODEL_PATH2 \
 --tokenizer_name $YOUR_MODEL_PATH2 \
 --dataset_name $YOUR_DATASET_PATH \
 --output_dir $DATA_SAVE_PATH2 \
 --tensor_parallel_size 8 \
 --model_dtype "float16" \
 --batch_size 50 \
 --shuffle False \
 --temperature 20 \
 --top_k 20 \
 --top_p 0.9 \
 --max_tokens 512 \
 --presence_penalty 1.0
```


### 3. Merge these two inference results and load reward model checkpoint to generate scores:
```shell
python data_process/merge.py \
 --dataset_name1 $DATA_SAVE_PATH1 \
 --dataset_name2 $DATA_SAVE_PATH2 \
 --output_dir $MERGED_DATA_SAVE_PATH \

accelerate launch --num_machines 1 --num_processes 1 inference/rm_inference.sh \
 --model_name $YOUR_REWARD_MODEL_PATH \
 --tokenizer_name $YOUR_REWARD_MODEL_PATH \
 --dataset_name $MERGED_DATA_SAVE_PATH \
 --output_dir $RM_LABELED_DATA_SAVE_PATH \
 --batch_size 50 \
 --shuffle False
```


### 4. Merge the reward model labeled dataset into pairs and filter it in a threshold value:
```shell
python data_process/get_pairs.py \
 --dataset_name $MERGED_DATA_SAVE_PATH \
 --output_dir $PAIR_DATA_SAVE_PATH \

python data_process/get_res.py \
 --dataset_name $PAIR_DATA_SAVE_PATH \
 --output_dir1 $RM_LOOP_TRAIN_DATA_SAVE_PATH \
 --output_dir2 $REMAINING_DATA_SAVE_PATH \
 --left_threshold 0.45 \
 --left_threshold 0.55 \
```


### 5. Training reward model:
```shell
export PRETRAIN_MODEL_SAVE_PATH = {PRETRAIN_MODEL_SAVE_PATH}
# the path to reward model lora weight to resume training
export REWARD_MODEL_LORA_SAVE_PATH = {REWARD_MODEL_LORA_SAVE_PATH}
export RM_LOOP_MODEL_SAVE_PATH = {RM_LOOP_MODEL_SAVE_PATH}

accelerate launch --multi_gpu --num_machines 1 --num_processes 8 rm_training/rm_training.py \
 --model_name $PRETRAIN_MODEL_SAVE_PATH \
 --tokenizer_name $PRETRAIN_MODEL_SAVE_PATH \
 --resume_from_checkpoint $REWARD_MODEL_LORA_SAVE_PATH \
 --dataset_name $RM_LOOP_TRAIN_DATA_SAVE_PATH \
 --load_from_json True \
 --eval_subset "data/evaluation" \
 --eval_subset 1500 \
 --output_dir $RM_LOOP_MODEL_SAVE_PATH \
 --eval_first_step False \
 --per_device_train_batch_size 5 \
 --per_device_eval_batch_size 1 \
 --gradient_accumulation_steps 1 \
 --learning_rate 2e-5 \
 --weight_decay 0.0 \
 --bf16 True \
 --num_train_epochs 5 \
 --gradient_checkpointing False \
 --optim "adamw_hf" \
 --lr_scheduler_type "linear" \
 --max_length "512" \
 --evaluation_strategy "steps" \
 --eval_steps 2500 \
 --save_strategy "steps" \
 --save_steps 2500 \
 --remove_unused_columns False \
 --label_names "[]" \
 --logging_strategy "steps" \
 --logging_steps 10 \
 --report_to "wandb" 
```