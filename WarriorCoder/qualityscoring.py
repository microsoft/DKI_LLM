from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,help='模型路径')
parser.add_argument('--start', type=int,help='开始')
parser.add_argument('--end', type=int,help='终止')
args = parser.parse_args()

name = args.path[args.path.rfind('/')+1:]

f = open("/blob/huawen/traindata/filtered_all_processed_instructions.json", 'r+')
fw = open(f"/blob/huawen/traindata/filtered_all_processed_instructions_score_{name}.json", 'w+')
lines = f.readlines()[args.start:args.end]

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(args.path, tensor_parallel_size=8, trust_remote_code=True, enforce_eager=True, max_model_len=8192)
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=8192)

system_prompt = "We are interested in understanding the quality of the following user query. Your task is to assess each prompt based on its clarity, specificity, coherence and difficulty."

for line in tqdm(lines):
    d = json.loads(line)
    instruction = d["instruction"]
    user_prompt = f"For each query, carry out the following steps:\n1. Assess the Quality: Evaluate whether the query is a clear instruction without a final response and consider the level of difficulty. It is acceptable for a query to include a response that needs improvement or correction and ask the AI to enhance it. Provide a brief explanation of your reasoning.\n2. Assign a Score: Rate the query on a scale of 1 to 10, where a higher score reflects higher quality.\nGuidelines for Scoring:\n• Execellent (9-10): For queries that are very clear, specific, and well-articulated, without a final response included. These queries are particularly challenging and excellently designed to assess the AI's proficiency.\n• Good (5-8): For queries that are clear and specific instructions without a final response. These are not overly difficult to answer and moderately assess the AI's capabilities.\n• Average (3-4): For queries that are fairly clear and specific instructions without a final response. These queries are easy to answer.\n• Poor (1-2): For queries that are ambiguous or include the final response within the instruction itself.\nEnsure to critically evaluate the user query and avoid giving high scores to the query that is ambiguous or too long.\n\n[User Question]\n{instruction}\n\n[Output Format]\nUse double square brackets to format your scores, like so: [[5]]."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    outputs = llm.generate(prompts=[text], sampling_params=sampling_params)
    d["quality_judgement"] = outputs[0].outputs[0].text
    print("=====================================")
    print(instruction)
    print("########################################")
    print(d["quality_judgement"])
    print("=====================================")
    fw.write(json.dumps(d)+"\n")