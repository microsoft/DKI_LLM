from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline,get_scheduler,AutoModelForSequenceClassification,AutoModelForCausalLM,AutoTokenizer,HfArgumentParser
from trl import AutoModelForCausalLMWithValueHead, ModelConfig, PPOv2Config, PPOv2Trainer, set_seed
from trl.core import LengthSampler
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


tqdm.pandas()
wandb.login(key='')

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="", metadata={"help": "the subset to use"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.4e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    input_max_length: Optional[int] = field(default=512, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the learning rate scheduler type"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.5,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=True, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=200, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "the outpur dir"})
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})
    steps: Optional[int] = field(default=20000, metadata={"help": "number of epochs"})
    epochs: Optional[int] = field(default=4, metadata={"help": "number of epochs"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    local_rank: Optional[int] = field(default=0, metadata={"help": "local rank"})
    load_from_json: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to load dataset from json file"},
    )
    reward_score_normaliaztion: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use the reward score normaliaztion"},
    )
    resume_epoch: Optional[int] = field(default=0, metadata={"help": "resume epoch"})
    resume_steps: Optional[int] = field(default=0, metadata={"help": "resume steps"})



parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
reward_model_name = script_args.reward_model_name





if script_args.load_from_json == True:
    train_dataset = load_dataset('json', data_files=script_args.dataset_name, split="train")
else:
    train_dataset = load_dataset(script_args.dataset_name, data_dir=script_args.subset, split="train")


# We then define the arguments to pass to the sentiment analysis pipeline.
# We set `return_all_scores` to True to get the sentiment score for each token.
sent_kwargs = {
    "return_all_scores": True,
    #"function_to_apply": "none",
    "batch_size": script_args.batch_size,
    "truncation": True,
}


tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


# GPT-2 tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(ds, tokenizer):
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(example):
        #for question in examples["question"]:
        query = "Question: " + example["question"] + "\n\nAnswer: "
        tokenized_question = tokenizer(query,padding=False)
        #new_examples["query"].append(query)
        #new_examples["input_ids"].append(tokenized_question["input_ids"])

        return tokenized_question


    return ds.map(preprocess_function, remove_columns=original_columns,
                  num_proc=num_proc)


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(train_dataset, tokenizer)
dataset  = dataset.filter(lambda x: len(x["input_ids"]) < 512, num_proc=24)
dataset.set_format(type="torch")

script_args.steps = int(len(dataset) / script_args.batch_size) * script_args.ppo_epochs


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


config = PPOv2Config(
    #steps=script_args.steps,
    num_train_epochs=script_args.epochs,
    sft_model_path=script_args.model_name,
    learning_rate=script_args.learning_rate,
    report_to=script_args.log_with,
    per_device_train_batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    #optimize_cuda_cache=True,
    #early_stopping=script_args.early_stopping,
    #target_kl=script_args.target_kl,
    num_ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    #init_kl_coef=script_args.init_kl_coef,
    #adap_kl_ctrl=script_args.adap_kl_ctrl,
    exp_name="self_rlaif",
    output_dir = script_args.output_dir,
    bf16=script_args.bf16,
    lr_scheduler_type=script_args.lr_scheduler_type,
    response_length= script_args.output_max_length,
    stop_token_id = tokenizer.eos_token_id,
    save_steps=20,
    auto_find_batch_size = True
    #tracker_kwargs={"wandb": {"name": script_args.output_dir}}
)



def score_normalization(score):
    if score < script_args.reward_baseline:
        score = script_args.reward_baseline
    return (score - script_args.reward_baseline) * (1 - -1) / (1 - script_args.reward_baseline) + -1



def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

current_device = Accelerator().local_process_index
model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name, num_labels=1,
    #load_in_8bit=True,
    #device_map={"": current_device},
    #peft_config=lora_config,
)

model = get_peft_model(model, peft_config)

if model.config.pad_token_id is None:
    model.config.pad_token_id=tokenizer.pad_token_id

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name, num_labels=1
)

if reward_model.config.pad_token_id is None:
    reward_model.config.pad_token_id=tokenizer.pad_token_id

ref_policy = AutoModelForCausalLM.from_pretrained(
    script_args.model_name
)

policy = AutoModelForCausalLM.from_pretrained(
    script_args.model_name
)

policy = get_peft_model(policy, lora_config)

'''
optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )'''




optimizer = Adafactor(
    filter(lambda p: p.requires_grad, model.parameters()),
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    lr=config.learning_rate,
)

lr_scheduler = get_scheduler(
    name=script_args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=script_args.steps,
)


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOv2Trainer(
    config,
    value_model=model,
    reward_model=reward_model,
    policy =policy,
    ref_policy = ref_policy,
    #ref_model=None,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=collator,
    #optimizers=optimizer,
    #lr_scheduler=lr_scheduler
)






generation_kwargs = {
    # "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)


ppo_trainer.train()
ppo_trainer.save_model(script_args.output_dir)
ppo_trainer.save_state()
