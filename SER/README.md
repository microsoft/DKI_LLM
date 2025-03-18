# Self-Evolve Reward Learning for LLMs

[![Project](http://img.shields.io/badge/Project-SER-E3E4C8.svg)](https://microsoft.github.io/DKI_LLM/)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2411.00418-99D4C8.svg)](https://arxiv.org/abs/2411.00418)

In this paper, we propose Self-Evolved Reward Learning (SER), a novel approach where the RM generates additional training data to iteratively improve itself. We conducted extensive experiments on multiple datasets such as HH-RLHF and UltraFeedback, using models like Mistral and Llama 3, and compare SER against various baselines. Our results demonstrate that even with limited human-annotated data, learning from self-feedback can robustly enhance RM performance, thereby boosting the capabilities of large language models (LLMs).

<div align="center">
  <img width="70%" src="docs/overview.png">
</div>

## Quick Start 🚀

### Step 1: Build Environment
```bash
conda env create -f environment.yaml
conda activate ser

```

### Step 2: Prepare Data and Checkpoints
- Download raw data from [Ultrafeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), [Summarize](https://huggingface.co/datasets/HuggingFaceH4/summarize-from-feedback), [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), [Stackoverflow](https://huggingface.co/datasets/HuggingFaceH4/stack-exchange-preferences).
- Download base model from [LLama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B),[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1), [LLama2-13B](https://huggingface.co/meta-llama/Llama-2-13b), [LLama2-70B](https://huggingface.co/meta-llama/Llama-2-70b)

### Step 3: Prepare SFT models

Our reward model is initialized from the SFT model. We initialize the base model and execute the following script:

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

### Step 4: Train the SER loops
We use the following script to train the reward model:
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

we use the following script to filter pair-wise data:

```shell
python  rm_loop_filter_by_self.py \
 --dataset_name {YOUR_DATASET_PATH} \
 --rm_next_loop_save_path {USEFUL_DATA_SAVE_PATH} \
 --remaining_data_save_path {UNUSEFUL_DATA_SAVE_PATH} \
 --left_threshold {UNUSEFUL_DATA_SAVE_PATH} \
```


### Step 5: Train the Policy Model
After obtaining the reward model, we use sft model as the base policy model for training:
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



## Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@article{huang2024self,
  title={Self-Evolved Reward Learning for LLMs},
  author={Huang, Chenghua and Fan, Zhizhen and Wang, Lu and Yang, Fangkai and Zhao, Pu and Lin, Zeqi and Lin, Qingwei and Zhang, Dongmei and Rajmohan, Saravan and Zhang, Qi},
  journal={arXiv preprint arXiv:2411.00418},
  year={2024}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Question

If you have any question or find any bug, please go ahead and [open an issue](https://github.com/microsoft/DKI_LLM/issues). Issues are an acceptable discussion forum as well.

If you want to concat the author, please email: `huangch22 AT m.fudan.edu.cn`.