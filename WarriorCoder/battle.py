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

fw = open(f"modela_battle_modelb_qwen_{args.start}_{args.end}.json", 'w+')

prompts = []


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.path, trust_remote_code=True)
f1 = open("", "r+")
f2 = open("", "r+")
lines1 = f1.readlines()[args.start:args.end]
lines2 = f2.readlines()[args.start:args.end]
t = 0
for line1, line2 in zip(lines1, lines2):
    d1 = json.loads(line1)
    d2 = json.loads(line2)
    instruction = d1["messages"][0]["content"]
    answer1 = d1["messages"][1]["content"]
    answer2 = d2["messages"][1]["content"]
    #print(answer1)
    #print(answer2)
    if t%2 == 0:
        prompt = f"This is a chatbot arena. You will be given assistant A’s answer, and assistant B’s answer. Please act as an impartial judge and evaluate the capability of two AI assistants. You should choose the assistant that follows instructions and answers questions better. Your evaluation should consider factors such as helpfulness, relevance, and accuracy. Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. DO NOT allow the LENGTH of the responses to influence your evaluation, choose the one that is straight-to-the-point instead of unnecessarily verbose. When the two candidates perform equally well, choose the SHORTER answer. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation concisely within 200 words, output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better, and “[[Tie]]” for a tie. Finish your judgement within 300 words.\n\n[User Question]\n{instruction}\n\n[The Start of Assistant A’s Answer]\n{answer1}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer2}\n[The End of Assistant B’s Answer]"
    else:
        prompt = f"This is a chatbot arena. You will be given assistant A’s answer, and assistant B’s answer. Please act as an impartial judge and evaluate the capability of two AI assistants. You should choose the assistant that follows instructions and answers questions better. Your evaluation should consider factors such as helpfulness, relevance, and accuracy. Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. DO NOT allow the LENGTH of the responses to influence your evaluation, choose the one that is straight-to-the-point instead of unnecessarily verbose. When the two candidates perform equally well, choose the SHORTER answer. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation concisely within 200 words, output your final verdict by strictly following this format: “[[A]]” if assistant A is better, “[[B]]” if assistant B is better, and “[[Tie]]” for a tie. Finish your judgement within 300 words.\n\n[User Question]\n{instruction}\n\n[The Start of Assistant A’s Answer]\n{answer2}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer1}\n[The End of Assistant B’s Answer]"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    prompts.append(text)
    t = t + 1
    
# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(args.path, dtype="float16", tensor_parallel_size=8, trust_remote_code=True, max_model_len=8192, enforce_eager=True)
sampling_params = SamplingParams(temperature=1.0, top_p=0.995, max_tokens=8192)
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
t = 0
for output in outputs:
    d = {"arena": output.outputs[0].text, "t": t}
    t = t + 1
    fw.write(json.dumps(d)+"\n")
