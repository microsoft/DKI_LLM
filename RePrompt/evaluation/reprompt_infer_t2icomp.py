import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch.distributed as dist
import math

import torch
from torch import autocast, nn
from diffusers import DiffusionPipeline
from diffusers import FluxPipeline
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import ImageReward as RM
import json

from tqdm import tqdm
from diffusers import Transformer2DModel, PixArtSigmaPipeline


model_path = "/output/reprompt_with_flux/checkpoint-100"
base = False
outdir = "output/results/reprompt_flux_on_t2icompbench"
num_images_per_prompt = 10
sd_model = "flux"

dist.init_process_group("nccl")
rank = dist.get_rank()
device = rank % torch.cuda.device_count()

torch.cuda.set_device(device)

llm_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if sd_model == "flux":
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16,
                                                        ).to(device)
elif sd_model == "sd3":
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16,
                                                ).to(device)
elif sd_model == "sdxl":
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
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


imagereward_model = RM.load("ImageReward-v1.0").to(device)

dist.barrier()
global_n_samples = dist.get_world_size()
    # Load prompts
filenames = []

for filename in os.listdir("evaluation/T2I-CompBench/examples/dataset"):
    if filename.endswith("_val.txt"):
        filenames.append(filename[:-len("_val.txt")])

# print(filenames)
for filename in filenames:
    full_path = os.path.join("evaluation/T2I-CompBench/examples/dataset", filename + "_val.txt")
    with open(full_path, 'r') as f:
        lines = f.readlines()
        metadatas = [{"prompt": line.strip()} for line in lines]

    total_prompts = int(math.ceil(len(metadatas) / global_n_samples) * global_n_samples)
    new_metadatas = metadatas + [metadatas[0]] * (total_prompts - len(metadatas))
    per_gpu_prompts = new_metadatas[rank:total_prompts:global_n_samples]

    for index, metadata in tqdm(enumerate(per_gpu_prompts), total=len(per_gpu_prompts), desc=f"Rank {rank}"):
        global_index = index * global_n_samples + rank

        if global_index > len(metadatas) - 1:
            break

        prompt = metadata['prompt']

        sample_path = os.path.join(outdir, filename, "samples")
        os.makedirs(sample_path, exist_ok=True)

        v = prompt
        message = [{"role": "system",
                    "content": "You are Prompt Wizard assistant, designed to act as the ultimate creative muse for image generation model users. "
                               "Your core purpose is to translate user request into an effective, detailed, imaginative, and optimized prompt that unlock the full potential of image generation model. "
                               "You can reasoning style, subject's characteristics, background details, and interactions with color and lighting of the image."
                               "The reasoning process and final prompt are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> a photo of detail </answer>."},
                   {"role": "user", "content": "{}".format(v)},
                   {"role": "assistant", "content": "Let me solve this step by step.\n<think>"}]
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = llm_model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]

        diff = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # print(diff)
        metadata["response"] = diff


        match = re.search(r'<answer>(.*?)</answer>', diff)
        if match:
            sample = match.group(1)

        else:
            sample = 'a photo'

        if base:
            sample = prompt

        metadata['final_prompt'] = sample

        generator = [torch.Generator(device="cuda").manual_seed(42 + i - 1) for i in range(num_images_per_prompt)]


        if sd_model == "flux":
            sub_images = pipe(sample,
                                num_images_per_prompt=num_images_per_prompt,
                                height=512,
                                width=512,
                                guidance_scale=3.5,
                                num_inference_steps=50,
                                max_sequence_length=512,
                                generator=generator).images
        elif sd_model == "sd3":
            with autocast("cuda"):
                sub_images = pipe(
                    sample,
                    negative_prompt="",
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator
                ).images
        elif sd_model == "sdxl":
            sub_images = pipe(
                sample,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images
        elif sd_model == "pixart":
            # with autocast("cuda"):
            sub_images = pipe(sample,
                              num_images_per_prompt=num_images_per_prompt).images

        for sample_count, image in enumerate(sub_images):
            image.save(os.path.join(sample_path, f"{prompt}_{sample_count:06}.png"))

dist.barrier()
if rank == 0:
    print("Done.")
dist.barrier()
dist.destroy_process_group()

