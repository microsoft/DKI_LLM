import json
import os
import numpy as np


res1_path = "data/90k_pairs_labeling.json"
save_path = "data/90k_pairs_before_labeling.json"


with open(res1_path, 'r') as file:
    res1 = file.read()

res1 = json.loads(res1)


for i in range(len(res1)):
    with open(save_path, 'a') as f:
        f.write(
            json.dumps(
                dict(
                    question=res1[i]["question"],
                    answer=res1[i]["answer_epoch1"],
                ),
                indent=4
            )
        )
        f.write(
            json.dumps(
                dict(
                    question=res1[i]["question"],
                    answer=res1[i]["answer_epoch10"],
                ),
                indent=4
            )
        )

