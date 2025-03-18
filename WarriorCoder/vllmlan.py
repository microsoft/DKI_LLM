from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,help='模型路径')
parser.add_argument("--n", type=int, default=200, help="Number of samples to generate for one time.")
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--repeat", type=int, default=None, help="Number of times to repeat the instruction generation. Only available when total prompts is not specified.")
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument('--language', type=str,help='语言')
args = parser.parse_args()

name = args.path[args.path.rfind('/')+1:]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)


# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(args.path, tensor_parallel_size=8, trust_remote_code=True)
stop_tokens = ["<|eot_id|>","<|end_of_text|>","<|starter_header_id|>","<|end_header_id|>"]
sampling_params = SamplingParams(
    n=args.n,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    stop=stop_tokens)

# Prepare your prompts
text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant designed to provide helpful, step-by-step guidance on {args.language} coding problems. The user will ask you a wide range of {args.language} coding questions.\nYour purpose is to assist users in understanding {args.language} coding concepts, working through {args.language} code, and arriving at the correct {args.language} solutions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
fw = open("/blob/huawen/generated_instructions/" + name + "_t_" + str(args.temperature) + "_p_" + str(args.top_p) + "_" + args.language + "_output.jsonl", 'w+')
# generate outputs
for i in range(args.repeat):
    outputs = llm.generate(prompts=[text], sampling_params=sampling_params)
    for output in outputs:
        print(output.outputs[0].text)
        fw.write(json.dumps({"instruction": output.outputs[0].text})+"\n")
# Print the outputs.
