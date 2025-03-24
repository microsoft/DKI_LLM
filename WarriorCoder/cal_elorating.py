import argparse
import json
from tqdm import tqdm

f = open("", "r+")
lines = f.readlines()

athene_elo = 0
deepseekcoder_elo = 0
llama_elo = 0
qwen_elo = 0
qwq_elo = 0

def update_elo(A_elo, B_elo, A_score, B_score):
    if A_score>B_score:
        A_elo = A_elo + 40 * (1.-1./(1.+10**((B_elo-A_elo)/400)))
        B_elo = B_elo + 40 * (0.-1./(1.+10**((A_elo-B_elo)/400)))
        return A_elo, B_elo
    if A_score==B_score:
        A_elo = A_elo + 40 * (0.5-1./(1.+10**((B_elo-A_elo)/400)))
        B_elo = B_elo + 40 * (0.5-1./(1.+10**((A_elo-B_elo)/400)))
        return A_elo, B_elo
    if A_score<B_score:
        A_elo = A_elo + 40 * (0.-1./(1.+10**((B_elo-A_elo)/400)))
        B_elo = B_elo + 40 * (1.-1./(1.+10**((A_elo-B_elo)/400)))
        return A_elo, B_elo

for line in tqdm(lines):
    d = json.loads(line)
    curscore = {}
    for score, name in zip(d["scorelist"], d["modelnames"]):
        curscore[name] = score
    curscore[d["judgename"]] = 5.5
    athene_elo, deepseekcoder_elo = update_elo(athene_elo, deepseekcoder_elo, curscore["athene"], curscore["deepseekcoder"])
    athene_elo, llama_elo = update_elo(athene_elo, llama_elo, curscore["athene"], curscore["llama"])
    athene_elo, qwen_elo = update_elo(athene_elo, qwen_elo, curscore["athene"], curscore["qwen"])
    athene_elo, qwq_elo = update_elo(athene_elo, qwq_elo, curscore["athene"], curscore["qwq"])
    deepseekcoder_elo, llama_elo = update_elo(deepseekcoder_elo, llama_elo, curscore["deepseekcoder"], curscore["llama"])
    deepseekcoder_elo, qwen_elo = update_elo(deepseekcoder_elo, qwen_elo, curscore["deepseekcoder"], curscore["qwen"])
    deepseekcoder_elo, qwq_elo = update_elo(deepseekcoder_elo, qwq_elo, curscore["deepseekcoder"], curscore["qwq"])
    llama_elo, qwen_elo = update_elo(llama_elo, qwen_elo, curscore["llama"], curscore["qwen"])
    llama_elo, qwq_elo = update_elo(llama_elo, qwq_elo, curscore["llama"], curscore["qwq"])
    qwen_elo, qwq_elo = update_elo(qwen_elo, qwq_elo, curscore["qwen"], curscore["qwq"])
print(athene_elo)
print(deepseekcoder_elo)
print(llama_elo)
print(qwen_elo)
print(qwq_elo)
