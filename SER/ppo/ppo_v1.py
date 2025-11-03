from dataclasses import dataclass, field
from typing import Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline,get_scheduler
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
import wandb

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
    output_max_length: Optional[int] = field(default=1024, metadata={"help": "maximum length for generation"})
    input_max_length: Optional[int] = field(default=1024, metadata={"help": "maximum length for generation"})
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
    #"function_to_apply": "",
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

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["question"]:
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) < script_args.input_max_length, batched=False)

    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(train_dataset, tokenizer)

script_args.steps = int(len(dataset) / script_args.batch_size) * script_args.ppo_epochs


config = PPOConfig(
    steps=script_args.steps,
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
    init_kl_coef=script_args.init_kl_coef,
    adap_kl_ctrl=script_args.adap_kl_ctrl,
    tracker_project_name="self_rlaif",
    tracker_kwargs={"wandb": {"name": script_args.output_dir}}
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


current_device = Accelerator().local_process_index
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    #load_in_8bit=True,
    #device_map={"": current_device},
    peft_config=lora_config,
)

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
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler
)


# We then build the sentiment analysis pipeline using our reward model, passing the
# model name and the sentiment analysis pipeline arguments. Let's also make sure to
# set the device to the same device as the PPOTrainer.
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a ` pipeline` bug
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"torch_dtype": torch.bfloat16 if script_args.bf16 is True else torch.float16},
    tokenizer=tokenizer,
    return_token_type_ids=False,
)


# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
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


if script_args.reward_score_normaliaztion:
    for epoch_id in range(script_args.epochs):
        for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
            question_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute reward score (using the sentiment analysis pipeline)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            try:
                pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            except:
                print(f"Error at step {step} when computing rewards")
                continue

            rewards = [torch.tensor(score_normalization(output[0]["score"])) for output in pipe_outputs]

            try:
                stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)
            except:
                print(f"Error at step {step} when stepping")
                continue

            if step !=0 and step % script_args.save_freq == 0:
                ppo_trainer.save_pretrained(script_args.output_dir + f"epoch_{epoch_id}_step_{step}")
                print(f"epoch {epoch_id}, step {step}")
else:
    for epoch_id in range(script_args.epochs):
        for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
            question_tensors = batch["input_ids"]

            response_tensors = ppo_trainer.generate(
                question_tensors,
                return_prompt=False,
                length_sampler=output_length_sampler,
                **generation_kwargs,
            )
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            # Compute reward score (using the sentiment analysis pipeline)
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            try:
                pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            except:
                print(f"Error at step {step} when computing rewards")
                continue

            rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

            try:
                stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
                ppo_trainer.log_stats(stats, batch, rewards)
            except:
                print(f"Error at step {step} when stepping")
                continue

            if step !=0 and step % script_args.save_freq == 0:
                ppo_trainer.save_pretrained(script_args.output_dir + f"epoch_{epoch_id}_step_{step}")
                print(f"epoch {epoch_id}, step {step}")
