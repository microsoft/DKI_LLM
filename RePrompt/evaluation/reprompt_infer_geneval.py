import os
import csv
import pytorch_lightning as pl
#
pl.seed_everything(42)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch.distributed as dist
import math

import torch
from torch import autocast, nn
from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import json

from tqdm import tqdm
from torchvision.transforms import ToTensor
from einops import rearrange
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
from diffusers import Transformer2DModel, PixArtSigmaPipeline

model_path = "/output/reprompt_with_flux/checkpoint-100"
base = False
outdir = "output/results/reprompt_flux_on_geneval"
sd_model = "flux"
# Setup DDP:
dist.init_process_group("nccl")
rank = dist.get_rank()
device = rank % torch.cuda.device_count()

torch.cuda.set_device(device)

llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


if sd_model == "flux":
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16,
                                                        ).to(device)
elif sd_model == "sd3":
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16,
                                                ).to(device)
elif sd_model == "pixart":
    weight_dtype = torch.bfloat16
    transformer = Transformer2DModel.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        subfolder='transformer',
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        transformer=transformer,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    ).to(device)


# Load prompts
with open("evaluation/geneval/prompts/evaluation_metadata.jsonl") as fp:
    metadatas = [json.loads(line) for line in fp]

dist.barrier()
global_n_samples = dist.get_world_size()
total_prompts = int(math.ceil(len(metadatas) / global_n_samples) * global_n_samples)
new_metadatas = metadatas + [metadatas[0]] * (total_prompts - len(metadatas))
per_gpu_prompts = new_metadatas[rank:total_prompts:global_n_samples]

for index, metadata in tqdm(enumerate(per_gpu_prompts), total=len(per_gpu_prompts), desc=f"Rank {rank}"):
    global_index = index * global_n_samples + rank

    if global_index > len(metadatas) - 1:
        break

    outpath = os.path.join(outdir, f"{global_index:0>5}")
    os.makedirs(outpath, exist_ok=True)

    prompt = metadata['prompt']
    print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    v = prompt

    message = [{"role":"system", "content": "You are Prompt Wizard assistant, designed to act as the ultimate creative muse for image generation model users. "
                                  "Your core purpose is to translate user request into an effective, detailed, imaginative, and optimized prompt that unlock the full potential of image generation model. "
                                  "You can reasoning style, subject's characteristics, background details, and interactions with color and lighting of the image."
                                  "The reasoning process and final prompt are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> a photo of detail </answer>."},
                                   {"role":"user", "content":"{}".format(v)},
                                   {"role":"assistant", "content":"Let me solve this step by step.\n<think>"}]

    text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = llm_model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)
    # generated_ids = llm_model.generate(model_inputs.input_ids, max_new_tokens=2048, do_sample=False)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]

    diff = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(diff)
    metadata["response"] = diff

    match = re.search(r'<answer>(.*?)</answer>', diff, re.DOTALL)
    if match:
        sample = match.group(1).strip()
    else:
        sample = 'a photo'

    if base:
        sample = prompt

    metadata['final_prompt'] = sample
    with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
        json.dump(metadata, fp)

    if sd_model == "flux":
        sub_images = pipe(sample,
                            num_images_per_prompt=4,
                            height=512,
                            width=512,
                            guidance_scale=3.5,
                            num_inference_steps=50,
                            max_sequence_length=512,).images
    elif sd_model == "sd3":
        with autocast("cuda"):
            sub_images = pipe(
                sample,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
                num_images_per_prompt=4,
            ).images
    elif sd_model == "sdxl":
        sub_images = pipe(
            sample,
            num_images_per_prompt=4,
        ).images
    elif sd_model == "pixart":
        # with autocast("cuda"):
        sub_images = pipe(sample,
                          num_images_per_prompt=4).images

    all_samples = list()
    for sample_count, image in enumerate(sub_images):
        image.save(os.path.join(sample_path, f"{sample_count:05}.png"))

    all_samples.append(torch.stack([ToTensor()(sample) for sample in sub_images], 0))
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    grid = Image.fromarray(grid.astype(np.uint8))
    grid.save(os.path.join(outpath, f'grid.png'))
    del grid
    del all_samples
dist.barrier()
if rank == 0:
    print("Done.")
dist.barrier()
dist.destroy_process_group()

