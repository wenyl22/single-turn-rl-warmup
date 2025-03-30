API_KEYS = ["sk-3ee8fe0163b54cdaad0989072a6126b7", "sk-04b6dc46201a4d08a3d916b623c0e358", 
"sk-cbce3b5ed161462a9d21ff5fafbed5c4", "sk-0557a61e247c424a955be81163e49b69",
"sk-62f7309ab6e84ea387d646159eac7bc7", "sk-25c76efa199643f88645a3e624bbbd9c",
"sk-0d0b5631acdb4736b9ca0607bceefff2", "sk-3d12683dcbd6496bb37a4ccd0f3fe0e2"]

from openai import OpenAI
import time
import glob
from datasets import load_dataset
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

eval_dataset = load_dataset('parquet', data_files={'eval': 'data/Freeway/DeepSeek-SFT/test.parquet'})['eval']

if not os.path.exists("data/sft"):
    os.makedirs("data/sft")
if not os.path.exists("data/sft/eval"):
    os.makedirs("data/sft/eval")
    
from itertools import cycle

api_key_cycle = cycle(API_KEYS)

def process_entry(i, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = eval_dataset[i]["prompt"]
    solution = eval_dataset[i]["solution"]
    available_actions = eval_dataset[i]["available_actions"]
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=prompt,
            temperature=0.6,
            top_p=0.95,
            max_tokens=8192
        )
        end_time = time.time()
        print(f"Time taken for index {i}: {end_time - start_time}")
        with open(f"data/sft/eval/{i}.jsonl", "w") as f:
            json.dump({
                "instruction": prompt[1]["content"],
                "system": prompt[0]["content"],
                "response_content": response.choices[0].message.content,
                "response_reasoning_content": response.choices[0].message.reasoning_content,
                "solution": solution,
                "available_actions": available_actions
            }, f)
    except Exception as e:
        print(f"Error processing index {i}: {e}")

if __name__ == "__main__":
    max_workers = len(API_KEYS)
    index = glob.glob("data/sft/eval/*.jsonl")
    index = [int(file.split("/")[-1].split(".")[0]) for file in index]
    print(f"Already processed {len(index)} entries")
    remain_index = set(range(len(eval_dataset))) - set(index)
    print(f"Remaining {len(remain_index)} entries")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_entry, i, next(api_key_cycle))
            for i in remain_index
        ]
        for future in as_completed(futures):
            future.result() 