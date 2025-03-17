import json
import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    output_dir: Optional[str] = field(default="", metadata={"help": "the output dir"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]


with open(script_args.dataset_name, 'r') as file:
    res1 = file.read()
res1 = json.loads(res1)


count1 = 0
length = int(len(res1) / 2)
for i in range(length):
    question1 = res1[i * 2]["question"].split("Question: ")[1].split("\n\nAnswer: ")[0]
    answer1 = res1[i * 2]["question"].split("\n\nAnswer: ")[1]
    reward1 = res1[i * 2]["reward"]

    question2 = res1[i * 2 + 1]["question"].split("Question: ")[1].split("\n\nAnswer: ")[0]
    answer2 = res1[i * 2 + 1]["question"].split("\n\nAnswer: ")[1]
    reward2 = res1[i * 2 + 1]["reward"]

    assert question1 == question2

    with open(script_args.output_dir, 'a') as f:
        f.write(
            json.dumps(
                dict(
                    question=question1,
                    response1=answer1,
                    response2=answer2,
                    reward1=reward1,
                    reward2=reward2,
                ),
            )
        )

            

