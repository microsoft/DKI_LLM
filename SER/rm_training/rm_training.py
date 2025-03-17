import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import transformers
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
from transformers.utils import PaddingStrategy
import matplotlib.pyplot as plt  # 新增
# torch.set_default_tensor_type(torch.cuda.HalfTensor)
import wandb
import huggingface_hub





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

    resume_training_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    train_subset: Optional[str] = field(default="", metadata={"help": "the subset to use"})
    eval_subset: Optional[str] = field(default="", metadata={"help": "the subset to use"})

    train_subset_num: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset_num: Optional[int] = field(
        default=1500,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    load_from_json: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to load dataset from json file"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to overwrite output dir"},
    )
    use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "use lora training"},
    )



parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

## load the hh-rlhf dataset from local

if data_args.load_from_json == True:
    train_dataset = load_dataset('json', data_files=data_args.dataset_name, split='train')
else:
    train_dataset = load_dataset(data_args.dataset_name, data_dir=data_args.train_subset, split="train")

eval_dataset = load_dataset('json', data_files=data_args.eval_subset, split='train')

'''没有可访问huggingface的环境，先注释了
# Load the human stack-exchange-paired dataset for tuning the reward model.
if data_args.load_from_json == True:
    train_dataset = load_dataset('json', data_files=data_args.dataset_name, split="train")
else:
    train_dataset = load_dataset(data_args.dataset_name, data_dir=data_args.train_subset, split="train")

eval_dataset = load_dataset(data_args.dataset_name, data_dir=data_args.eval_subset, split="train")'''

# Load the value-head model and tokenizer.
tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token


if "phi" in model_args.model_name or "Phi" in model_args.model_name:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules="all-linear"
    )
elif '70b' in model_args.model_name or '70B' in model_args.model_name:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=2,
        lora_alpha=32,
        lora_dropout=0.1
    )
else:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

config = AutoConfig.from_pretrained(model_args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name,
    #config=config,
    #trust_remote_code=True,
    num_labels=1,
    # torch_dtype=torch.float16
    torch_dtype=torch.bfloat16 if training_args.bf16 is True else torch.float16
)
if training_args.use_lora:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not training_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# Turn the dataset into pairs of post + summaries, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    if "question" in examples:
        for question, response_j, response_k in zip(examples["question"], examples["response_j"],
                                                    examples["response_k"]):
            if question is not None and response_j is not None and response_k is not None:
                tokenized_j = tokenizer("Question: " + question + "\n\nAnswer: " + response_j)
                tokenized_k = tokenizer("Question: " + question + "\n\nAnswer: " + response_k)

                new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
                new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
                new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
                new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])
            else:
                print(
                    f'Skipped example with None value: question="{question}", response_j="{response_j}", response_k="{response_k}"')
    else:
        print('Key error: "question" not in examples')

    return new_examples


# preprocess the dataset and filter out QAs that are longer than script_args.seq_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    # remove_columns=original_columns, ##用huggingface直接加载stackoverflow数据集需要这个，自己处理的hh-rlhf不需要
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= training_args.seq_length and len(
        x["input_ids_rejected"]) <= training_args.seq_length
)
'''自己划分数据集
if data_args.train_subset_num > 0:
    train_dataset = train_dataset.select(range(data_args.train_subset))'''

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    # remove_columns=original_columns,
)
eval_dataset = eval_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= training_args.seq_length and len(
        x["input_ids_rejected"]) <= training_args.seq_length
)

'''自己划分数据集
if data_args.eval_subset_num > 0:
    eval_dataset = eval_dataset.select(range(data_args.eval_subset))'''


@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    seq_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.seq_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.seq_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_j["input_ids"],
            "attention_mask_chosen": batch_j["attention_mask"],
            "input_ids_rejected": batch_k["input_ids"],
            "attention_mask_rejected": batch_k["attention_mask"],
            "return_loss": True,
        }
        #print(batch_j["input_ids"].shape)
        #print(batch_k["input_ids"].shape)
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("evaluate/metrics/accuracy")


# accuracy = evaluate.load("accuracy") 没huggingface环境

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        #print(len(inputs["input_ids_chosen"]))
        #print(len(inputs["input_ids_rejected"]))
        rewards_j = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_k = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_chosen": rewards_j, "rewards_rejected": rewards_k}
        return loss


# 自定义的Loss记录和绘图的callback
class LogLossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.train_losses.append((state.global_step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((state.global_step, logs["eval_loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        steps, train_losses = zip(*self.train_losses)
        _, eval_losses = zip(*self.eval_losses) if self.eval_losses else (steps, [])

        plt.figure()
        plt.plot(steps, train_losses, label="Training Loss")
        if eval_losses:
            plt.plot(steps, eval_losses, label="Evaluation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Evaluation Loss over Time")
        plt.savefig(os.path.join(args.output_dir, "loss_plot.png"))
        plt.show()


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, seq_length=training_args.seq_length),
)

# 添加callback
trainer.add_callback(LogLossCallback())

if training_args.eval_first_step:
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True


    trainer.add_callback(EvaluateFirstStepCallback())

# trainer.train()
trainer.train(model_args.resume_training_from_checkpoint)
trainer.save_model(training_args.output_dir)
trainer.save_state()
'''
print("Saving last checkpoint of the model")
model_name_split = model_args.model_name.split('/')[-1]
output_name = (
    f"{model_name_split}_peft_stack-exchange-paired_rmts__{data_args.train_subset_num}_{training_args.learning_rate}"
)
model.save_pretrained(output_name + "_peft_last_checkpoint")'''
