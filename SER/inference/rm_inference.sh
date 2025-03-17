accelerate launch --num_machines 1 --num_processes 1 rm_inference.py \
 --model_name {YOUR_MODEL_PATH} \
 --tokenizer_name {YOUR_TOKENIZER_PATH} \
 --dataset_name {YOUR_DATASET_PATH} \
 --output_dir {DATA_SAVE_PATH} \
 --batch_size 50 \
 --shuffle False