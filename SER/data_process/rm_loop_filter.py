import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    rm_next_loop_save_path: Optional[str] = field(default="", metadata={"help": "the output dir"})
    remaining_data_save_path: Optional[str] = field(default="", metadata={"help": "the output dir"})
    left_threshold: Optional[float] = field(default=0.45, metadata={"help": "the threshold to filter the dataset"})
    right_threshold: Optional[float] = field(default=0.55, metadata={"help": "the threshold to filter the dataset"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]


with open(script_args.dataset_name, 'r') as file:
    res = file.read()
res = json.loads(res)


use_list = []
remaining_list = []

for item in res:
    if (float(item['score_j'])>script_args.right_threshold) and (float(item['score_k'])<script_args.left_threshold):
        use_list.append(item)

    else:
        remaining_list.append(item)

if not os.path.exists(os.path.dirname(script_args.rm_next_loop_save_path)):
    os.makedirs(os.path.dirname(script_args.rm_next_loop_save_path))

with open(script_args.rm_next_loop_save_path, 'w') as f:
    json.dump(use_list, f, indent=2)

if not os.path.exists(os.path.dirname(script_args.remaining_data_save_path)):
    os.makedirs(os.path.dirname(script_args.remaining_data_save_path))

with open(script_args.remaining_data_save_path, 'w') as f:
    json.dump(remaining_list, f, indent=2)

