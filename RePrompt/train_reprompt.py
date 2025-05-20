

import re
import io

import argparse

import torch
from datetime import datetime
from PIL import Image
from accelerate import Accelerator
from accelerate import PartialState

from torch import autocast, nn
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline, Transformer2DModel, PixArtSigmaPipeline
from trl.trl import GRPOConfig, GRPOTrainer

# from CloudGPT_AOAI.cloudgpt_aoai import get_chat_completion, encode_image
import openai
import base64

openai.api_key = "your-api-key"

import ImageReward as RM

MASK_INDICES = [0, 1, 2]      # Indices of mask features in original list
MASK_FEATURE_MAP = {
    0: [22, 23, 24, 28, 29],      # 'body(mask)' masks related features 'body correct' & 'harmfulness'
    1: [25, 26],                  # 'face(mask)' masks related features 'face'
    2: [27],                      # 'hands(mask)' masks related features 'hands'
}


class PromptScorer:

    def __init__(self, sdmodel_name, args, accelerator):

        self.accelerator = accelerator
        # init scorer hparams
        self.lambda_aes = 0.05
        self.lambda_clip = 5.0
        self.num_images_per_prompt = 1

        # init models
        self.sdmodel_name = sdmodel_name
        self.init_diffusion_model()
        self.init_imagereward_model()

        self.eval_data_res = []

    def init_diffusion_model(self):
        if 'FLUX' in self.sdmodel_name:
            pipe = FluxPipeline.from_pretrained(self.sdmodel_name, torch_dtype=torch.bfloat16,
                                                )
        elif 'stable-diffusion-3' in self.sdmodel_name:
            pipe = StableDiffusion3Pipeline.from_pretrained(self.sdmodel_name, torch_dtype=torch.float16,
                                            )
        elif "PixArt" in self.sdmodel_name:
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
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(self.sdmodel_name, use_safetensors=True, variant="fp16")

        # Disable NSFW detect
        pipe.safety_checker = None
        self.diffusion_pipe = pipe

        self.distributed_state = PartialState()
        self.diffusion_pipe.to(self.distributed_state.device)

    def init_imagereward_model(self):
        self.imagereward_model = RM.load("ImageReward-v1.0")

    def gen_image_batched(self, prompts):
        images = []
        for pmpts in prompts:

            if 'FLUX' in self.sdmodel_name:
                sub_images = self.diffusion_pipe(
                    pmpts,
                    num_images_per_prompt=self.num_images_per_prompt,
                    height=512,
                    width=512,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,).images

            elif "stable-diffusion-3" in self.sdmodel_name:
                with autocast("cuda"):
                    sub_images = self.diffusion_pipe(
                        pmpts,
                        negative_prompt="",
                        num_inference_steps=28,
                        guidance_scale=7.0,
                        num_images_per_prompt=self.num_images_per_prompt,
                    ).images
            elif "PixArt" in self.sdmodel_name:
                # with autocast("cuda"):
                sub_images = self.diffusion_pipe(pmpts,
                                  num_images_per_prompt=self.num_images_per_prompt).images
            elif "stable-diffusion-xl" in self.sdmodel_name:
                sub_images = self.diffusion_pipe(
                    pmpts,
                    num_inference_steps=20,
                    num_images_per_prompt=self.num_images_per_prompt,
                ).images
            images.append(sub_images[0])

        return images


    def evaluate_gpt4v_reward(self, image: Image.Image, prompt: str) -> float:
        """
        Calls the GPT-4V API to evaluate the image with respect to the prompt.
        The evaluation focuses primarily on alignment with the prompt (correct object counts, attributes, entities, relationships)
        and secondarily on image authenticity.

        We assume the API returns a score out of 10, which we then normalize to [0, 1].
        """
        # Convert the PIL Image to a BytesIO stream


        buffered = io.BytesIO()
        image.save(buffered, format='PNG')
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_meta = f"data:image/{'PNG'.lower()};base64"

        # Define API endpoint and headers (update with your actual API key)
        # api_url = "https://api.openai.com/v1/chat/completions"
        # headers = {
        #     "Authorization": "Bearer YOUR_API_KEY",
        # }

        # Construct the messages for the API
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert image evaluator. Focus primarily on whether the image aligns with the prompt, "
                    "including correct object counts, attributes, entities, and relationships; secondarily, consider image authenticity.")

            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please rate this image on a scale of 0-10 (10 being perfect) and explain your reasoning. Please put your score in <score> score </score>. Prompt: {prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{image_meta},{base64_image}",
                            "detail": "low"
                        }
                    }

                ]
            }
        ]


        try:

            # response = get_chat_completion(
            #     engine="gpt-4-visual-preview",
            #     messages=messages,
            # )
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=100
            )

            result = response.choices[0].message.content

            match = re.search(r'<score>(.*?)</score>', result)
            score = float(match.group(1))
            normalized_score = (score - 5) / 5.0  # Normalize to [0, 1]
            return normalized_score
        except Exception as e:
            print("Error calling GPT-4V API:", e)
            return 0.0

    def get_score_batched(self, prompts, plain_texts, plain_aes_score=None, clip_score=False, visionreward_score=False, imagereward_score=True):
        images = self.gen_image_batched(prompts)

        if imagereward_score:
            final_scores = []
            for prompt, image in zip(plain_texts, images):
                score = self.imagereward_model.score(prompt, [image])

                gpt_score = self.evaluate_gpt4v_reward(image, prompt)
                score = 0.5 * score / 2.0 + 0.5 * gpt_score

                final_scores.append(score)

        return final_scores

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/data")
    parser.add_argument("--sdmodel_name", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--gpt_path", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--max_new_tokens", type=int, default=-1)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="Qwen2.5-3B-GRPO-FLUX-8k")

    args = parser.parse_args()
    return args


def main(args):
    accelerator = Accelerator()
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 1

    scorer = PromptScorer(args.sdmodel_name, args, accelerator)

    scorer = accelerator.prepare(scorer)

    # TODO maybe shard the data for parallel training

    dataset = [x.strip() for x in list(open(args.data,'r'))]
    # print(len(dataset))
    dataset = [{'prompt': [{"role":"system", "content": "You are Prompt Wizard assistant, designed to act as the ultimate creative muse for image generation model users. "
                          "Your core purpose is to translate user request into an effective, detailed, imaginative, and optimized prompt that unlock the full potential of image generation model. "
                          "You can reasoning style, subject's characteristics, background details, and interactions with color and lighting of the image."
                          "The reasoning process and final prompt are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> a photo of detail </answer>."},
                           {"role":"user", "content":"{}".format(v)},
                           {"role":"assistant", "content":"Let me solve this step by step.\n<think>"}]
                          , 'ori_prompt':v} for v in dataset]


    def reward_t2i(completions, ori_prompt, **kwargs):

        # TODO use plain texts here
        diffuser_prompts = []
        plain_texts = []
        for i, (prompt,sample) in enumerate(zip(ori_prompt, completions)):


            match = re.search(r'<answer>(.*?)</answer>', sample[0]['content'])
            if match:
                sample = match.group(1)
            else:
                sample = 'a photo'

            diffuser_prompts.append(sample)
            plain_texts.append(prompt)

        scores = scorer.get_score_batched(diffuser_prompts, plain_texts, plain_aes_score=None)

        return scores

    def format_reward_func(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r'Let me solve this step by step.\n<think>.*?</think>\n<answer>.*?</answer>$'
        completion_contents = [completion[0]['content'] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        # print(matches)
        return [1.0 if match else -1.0 for match in matches]

    def reward_len(completions, **kwargs):
        scores = []
        for completion in completions:
            match = re.search(r'<answer>(.*?)</answer>', completion[0]['content'])
            if match:
                sample = match.group(1)
                if len(sample.split()) > 15 and len(sample.split()) < 77:
                    scores.append(1.0)
                else:
                    scores.append(-1.0)
            else:
                scores.append(0.0)
        return scores


    training_args = GRPOConfig(output_dir=args.outdir, bf16=True, resume_from_checkpoint=args.resume,
                               num_train_epochs=3, num_generations=4,
                               learning_rate=2e-6, gradient_accumulation_steps=2,
                               logging_steps=1, save_steps = 100, save_total_limit=2)
    trainer = GRPOTrainer(
        model=args.gpt_path,
        reward_funcs=[reward_t2i, format_reward_func, reward_len],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    args = get_args()
    main(args)