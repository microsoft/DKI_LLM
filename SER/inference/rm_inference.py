from tqdm import tqdm
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import json
import os
from dataclasses import dataclass, field

TEMPLATE = (
    "Question: {question}\n\nAnswer: {answer}"
)


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    output_dir: Optional[str] = field(default="", metadata={"help": "the output dir"})
    batch_size: Optional[int] = field(default=50, metadata={"help": "the batch size"})
    shuffle: Optional[bool] = field(default=False, metadata={"help": "whether to shuffle"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

with open(script_args.dataset_name, 'r') as file:
    data_json = file.read()
data_dict = json.loads(data_json)

tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = TEMPLATE.format_map({
            'question': self.texts[idx]['question'],
            'answer': self.texts[idx]['response_k']
            # 'answer': self.texts[idx]['answer']
        })
        return text


# 构建自定义数据集实例
my_dataset = MyDataset(texts=data_dict)
dataloader = DataLoader(my_dataset, batch_size=script_args.batch_size, shuffle=script_args.shuffle)

sent_kwargs = {
    "return_all_scores": True,
    # "function_to_apply": "none",
    "batch_size": script_args.batch_size,
    "truncation": True,
}

current_device = Accelerator().local_process_index
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=script_args.model_name,
    device_map={"": current_device},
    model_kwargs={"torch_dtype": torch.bfloat16},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

output_list = []
for batch in tqdm(dataloader):
    outputs = sentiment_pipe(
        batch,
        **sent_kwargs,
    )
    # print(outputs)
    for idx, (query, output) in enumerate(tqdm(zip(batch, outputs))):
        print(output)
        reward = output[0]["score"]
        # reward = output["score"]
        # print(output)
        output_list.append({
            'question': query,
            'reward': reward
        })

with open(script_args.output_dir, 'w') as f:
    json.dump(output_list, f, indent=2)