## The project provide the code for:
1. Completion-based Instruction Mining
2. Deduplication and Difficulty Filtering
3. Embedding-based Compression
4. Code Generating
5. Calculation for Votes and Elo Rating
6. Training

## Create environments:
```shell
conda create -n data python=3.10 -y
conda activate data
pip install vllm
pip install datasketch
pip install scikit-learn

conda create -n train python=3.10 -y
conda activate train
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
cd warriortrain
python -m pip install .
pip install trl==0.12.0
pip install numpy==1.26.4
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```


## 1. Completion-based Instruction Mining
We use vllm to speed up the minging process:
```shell
conda activate data
python vllmlan.py --path [model_path] --n 1 --top_p [config_for_top_p] --temperature [config_for_temperature] --repeat [repeat_times] --max_tokens [max_length_of_context] --language [programming_language]
```
e.g.:
```shell
python vllmlan.py --path Qwen/Qwen2.5-72B-Instruct --n 1 --top_p 0.99 --temperature 1.2 --repeat 10000 --max_tokens 2048 --language Python
```
### Notice:
Different models have various chat templates, so you need to change it according to the file: tokenizer_config.json.

## 2. Deduplication and Difficulty Filtering
After merging the generated instructions, we use MinHash algorithm to deduplicate them:
```shell
conda activate data
python fhw_deduplication.py
```
Then we use external LLMs to evaluate the quality of the instructions:
```shell
conda activate data
python qualityscoring.py --path [model_path] --start [start_pos] --end [end_pos]
```
e.g.:
```shell
python qualityscoring.py --path meta-llama/Llama-3.3-70B-Instruct --start 0 --end 40000
```
## 3. Embedding-based Compression
We use KcenterGreedy algorithm to compress the filtered instructions:
```shell
conda activate data
python kcenter.py --start [start_pos] --end [end_pos]
```
e.g.:
```shell
python kcenter.py --start 0 --end 40000 & python kcenter.py --start 40000 --end 80000 & python kcenter.py --start 80000 --end 120000
```
## 4. Code Generating
We use vllm to speed up the code generation process:
```shell
conda activate data
python vllmans.py --path [model_path] --datapath [path_to_instructions]
```
## 5. Calculation for Votes and Elo Rating
We conduct pariwise evaluations based on uninvolved llms:
```shell
conda activate data
python battle.py --path [model_path] --datapath [path_to_instructions_with answers]
```
After collecting all the voting results, we can calculate the Elo Rating:
```shell
conda activate data
python cal_elorating.py --path [model_path] --datapath [path_to_instructions_with answers]
```
## 6. Training
Finally, we train the target model with the instructions with the best responses. The data format is:
```shell
{"messages": [{"content": instructions_in_natural_language, "role": "user"}, {"content": best_response, "role": "assistant"}]}
```
To reproduce the results in our paper, you need to fill in the training configuration in **recipes/accelerate_configs/deepspeed_zero3.yaml** and **recipes/config_sft_warrior.yaml**.
Then you can start your training process by:
```shell
cd warriortrain
conda activate train
accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft_warrior.py recipes/config_sft_warrior.yaml
```
### Notice:
You also need to rewrite the path to your training and test data in **scripts/run_sft_warrior.py**.



