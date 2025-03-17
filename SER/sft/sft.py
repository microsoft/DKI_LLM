import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Sequence
import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
import transformers
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoConfig
)
from transformers.utils import PaddingStrategy
import wandb
import huggingface_hub



#wandb.init(project='RLAIF', name='RLAIF')


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="",
        metadata={"help": "the model name or path"}
    )
    tokenizer_name: Optional[str] = field(
        default="",
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model"
        }
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="", metadata={"help": "the subset to use"})
    load_from_json: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to load dataset from json file"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})


IGNORE_INDEX = -100
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    use_fast=True,
    #trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
num_proc = 24  # Can adjust to be higher if you have more processors.
#config = AutoConfig.from_pretrained(model_args.model_name,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_args.model_name)
#Tell Trainer not to attempt DataParallel
model.is_parallelizable = True
model.model_parallel = True


def preprocess_function(examples):
    sources = []
    targets = []
    for question, response_j in zip(examples["question"], examples["response_j"]):
        source = f"Question: {question}\n\nAnswer: "
        target = f"{response_j}{tokenizer.eos_token}"

        sources.append(source)
        targets.append(target)

    tokenized_sources = tokenizer(sources, return_attention_mask=False)
    tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

    all_input_ids = []
    all_labels = []

    for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
        input_ids = torch.LongTensor(s + t)
        labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)
        assert len(input_ids) == len(labels)
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    results = {'input_ids': all_input_ids, 'labels': all_labels}
    return results


if data_args.load_from_json == True:
    train_dataset = load_dataset('json', data_files=data_args.dataset_name, split="train")
else:
    train_dataset = load_dataset(data_args.dataset_name, data_dir=data_args.subset, split="train")
original_columns = train_dataset.column_names
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) <= training_args.seq_length)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer)
)
model.config.use_cache = False

trainer.train()
trainer.save_model(training_args.output_dir)
trainer.save_state()
