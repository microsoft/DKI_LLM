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
    left_threshold: Optional[int] = field(default=0.45, metadata={"help": "the threshold to filter the dataset"})
    right_threshold: Optional[int] = field(default=0.55, metadata={"help": "the threshold to filter the dataset"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]


with open(script_args.dataset_name, 'r') as file:
    res = file.read()
res = json.loads(res)


rewards = []
for item in res:
    rewards.append(item['reward1'])
    rewards.append(item['reward2'])


# 计算百分位数
left = np.percentile(rewards, script_args.left_threshold)
right = np.percentile(rewards, script_args.right_threshold)


for i in range(len(res)):
    if (res[i]['reward1'] <= left or res[i]['reward1'] >= right) and (res[i]['reward2'] <= left or res[i]['reward2'] >= right):
        if res[i]['reward1'] < res[i]['reward2']:
            count += 1
            with open(script_args.output_dir1, 'a') as f:
                f.write(
                    json.dumps(
                        dict(
                            question=res[i]['question'],
                            response_j=res[i]['response2'],
                            response_k=res[i]['response1'],
                        ),
                        indent=4
                    )
                )
        elif res[i]['reward1'] > res[i]['reward2']:
            count += 1
            with open(script_args.output_dir1, 'a') as f:
                f.write(
                    json.dumps(
                        dict(
                            question=res[i]['question'],
                            response_j=res[i]['response1'],
                            response_k=res[i]['response2'],
                        ),
                        indent=4
                    )
                )
        else:
            with open(script_args.output_dir2, 'a') as f:
                f.write(
                    json.dumps(
                        dict(
                            question=res[i]['question'],
                            answer=res[i]['response1'],
                        ),
                        indent=4
                    )
                )
            with open(script_args.output_dir2, 'a') as f:
                f.write(
                    json.dumps(
                        dict(
                            question=res[i]['question'],
                            answer=res[i]['response2'],
                        ),
                        indent=4
                    )
                )
    else:
        with open(script_args.output_dir2, 'a') as f:
            f.write(
                json.dumps(
                    dict(
                        question=res[i]['question'],
                        answer=res[i]['response1'],
                    ),
                    indent=4
                )
            )
        with open(script_args.output_dir2, 'a') as f:
            f.write(
                json.dumps(
                    dict(
                        question=res[i]['question'],
                        answer=res[i]['response2'],
                    ),
                    indent=4
                )
            )