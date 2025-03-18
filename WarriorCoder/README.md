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
```shell
conda activate data
python vllmlan.py --path [model_path] --n 1 --top_p [config_for_top_p] --temperature [config_for_temperature] --repeat [repeat_times] --max_tokens [max_length_of_context] --language [programming_language]
```
e.g.,
```shell
python vllmlan.py --path Qwen/Qwen2.5-72B-Instruct --n 1 --top_p 0.99 --temperature 1.2 --repeat 10000 --max_tokens 2048 --language Python
```
### Notice:
Different models have various chat templates, so you need to change it according to the file: tokenizer_config.json.

## 1. Deduplication and Difficulty Filtering

