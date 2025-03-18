import json
from tqdm import tqdm
import datasets
from datasketch import MinHashLSH, MinHash

idx = 0
lsh = MinHashLSH(threshold=0.5, num_perm=128)
f = open("filtered_all.jsonl",'r+')
fw = open("filtered_all_processed.json",'w+')
lines = f.readlines()
for line in tqdm(lines):
    d = json.loads(line)
    minhash = MinHash(num_perm=128)
    for word in d['instruction'].replace('.','').replace('\n',' ').split():
        minhash.update(word.encode('utf-8'))
    lsh.insert(str(idx), minhash)
    idx = idx + 1


idx = 0
t = 0
for line in tqdm(lines):
    d = json.loads(line)
    minhash = MinHash(num_perm=128)
    for word in d['instruction'].replace('.','').replace('\n',' ').split():
        minhash.update(word.encode('utf-8'))
    result = lsh.query(minhash)
    if len(result) == 1 or all(int(sim_idx)<= idx for sim_idx in result):
        t = t + 1
        fw.write(line)
    idx = idx + 1

print(t)
print(idx)



