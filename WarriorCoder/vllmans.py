from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,help='模型路径')
parser.add_argument('--datapath', type=str,help='模型路径')
args = parser.parse_args()

name = args.path[args.path.rfind('/')+1:]
dataname = args.datapath[args.datapath.rfind('/')+1:]
dataname = dataname.split("_")[0]
print(dataname)
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)


# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(args.path, tensor_parallel_size=8, trust_remote_code=True, enforce_eager=True, max_model_len=8192)
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=8192)


# Prepare your prompts
system_prompt = "You are an AI assistant designed to provide helpful, step-by-step guidance on Python coding problems. The user will ask you a wide range of Python coding questions. Your purpose is to assist users in understanding Python coding concepts, working through Python code, and arriving at the correct Python solutions."
f = open(f"{args.datapath}", 'r+')
lines = f.readlines()
fw = open(f"/cosmos/fhw/zsw/filterans_answerby/{dataname}_answerby_{name}.json", 'w+')
prompts = []
for line in tqdm(lines):
    d = json.loads(line)
    instruction = d["prompt"]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    prompts.append(text)
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
for line, output in zip(lines, outputs):
    d =json.loads(line)
    d["response"] = output.outputs[0].text
    fw.write(json.dumps(d)+"\n")
