import os
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Any, Dict, List, Optional, Union
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from tqdm import tqdm
import json
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field

print(torch.cuda.current_device())

TEMPLATE = (
    "Question: {question}\n\nAnswer: "
)


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    output_dir: Optional[str] = field(default="", metadata={"help": "the output dir"})
    tensor_parallel_size: Optional[int] = field(
        default=8, metadata={"help": "the number of GPUs"}
    )
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )
    batch_size: Optional[int] = field(default=50, metadata={"help": "the batch size"})
    shuffle: Optional[bool] = field(default=False, metadata={"help": "whether to shuffle"})
    temperature: Optional[float] = field(default=0.2, metadata={"help": "the temperature"})
    top_k: Optional[int] = field(default=20, metadata={"help": "the top_k"})
    top_p: Optional[float] = field(default=0.9, metadata={"help": "the top_p"})
    max_tokens: Optional[int] = field(default=512, metadata={"help": "the max tokens"})
    presence_penalty: Optional[float] = field(default=1.0, metadata={"help": "the presence penalty"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

with open(script_args.dataset_name, 'r') as json_file:
    data = json.load(json_file)
#data_dict = json.loads(data_json)


tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
config = AutoConfig.from_pretrained(script_args.model_name, trust_remote_code=True)
model_chosen = LLM(
    model=script_args.model_name,
    tokenizer=tokenizer_name,
    tokenizer_mode='auto',
    dtype=script_args.model_dtype,
    tensor_parallel_size=script_args.tensor_parallel_size,
    enable_chunked_prefill=False
    #config=config,
    #trust_remote_code=True
)


class MyDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = TEMPLATE.format_map({'question': self.texts[idx]['question']})
        return text


my_dataset = MyDataset(texts=data)
dataloader = DataLoader(my_dataset, batch_size=script_args.batch_size, shuffle=script_args.shuffle)

generation_config = dict(
    temperature=script_args.temperature,
    top_k=script_args.top_k,
    top_p=script_args.top_p,
    max_tokens=script_args.max_tokens,
    presence_penalty=script_args.presence_penalty,
)

output_list = []
final_output_list = []

for batch in tqdm(dataloader):
    outputs = model_chosen.generate(
        batch,
        SamplingParams(**generation_config),
        use_tqdm=False,
    )

    for idx, (query, output) in enumerate(tqdm(zip(batch, outputs))):
        output = output.outputs[0].text
        output_list.append(dict(question=query, answer=output))
        #with open(script_args.output_dir, 'a') as f:
        #f.write(json.dumps(dict(question=query, answer_chosen=output)))

for idx in range(len(output_list)):
    query = output_list[idx]['question']
    response = output_list[idx]['answer']

    output_response = ''.join(response.split("\n\nAnswer:")[-1]).strip()
    question_str = ''.join(query.split("Question: ")[-1]).strip()
    question = ''.join(question_str.split("\n\nAnswer:")[0]).strip()

    final_output_list.append({
        'question': question,
        'response': output_response,
    })
if not os.path.exists(os.path.dirname(script_args.output_dir)):
    os.makedirs(os.path.dirname(script_args.output_dir))
with open(script_args.output_dir, 'w') as f:
    json.dump(final_output_list, f, indent=2)
