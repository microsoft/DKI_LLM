import json
import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    dataset_name1: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    dataset_name2: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    output_dir: Optional[str] = field(default="", metadata={"help": "the output dir"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]


with open(script_args.dataset_name1, 'r') as file1:
    res1 = file1.read()
res1 = json.loads(res1)


with open(script_args.dataset_name2, 'r') as file2:
    res2 = file2.read()
res2 = json.loads(res2)


count = 0
for i in range(len(res1)):
    question1 = res1[i]["question"].split("Question: ")[1].split("\n\nAnswer: ")[0]
    answer1 = res1[i]["answer"]

    question2 = res2[i]["question"].split("Question: ")[1].split("\n\nAnswer: ")[0]
    answer2 = res2[i]["answer"]

    assert question1 == question2

    if answer1 != answer2:
        count += 1
        with open(script_args.output_dir, 'a') as f:
            if answer1[0] == ' ':
                f.write(
                    json.dumps(
                        dict(
                            question=question1,
                            answer=answer1[1:],
                        ),
                    )
                )
            else:
                f.write(
                    json.dumps(
                        dict(
                            question=question1,
                            answer=answer1,
                        ),
                    )
                )
        
            if answer2[0] == ' ':
                f.write(
                    json.dumps(
                        dict(
                            question=question1,
                            answer=answer2[1:],
                        ),
                    )
                )
            else:
                f.write(
                    json.dumps(
                        dict(
                            question=question1,
                            answer=answer2,
                        ),
                    )
                )