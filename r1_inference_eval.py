API_KEY = "sk-cdc26b886bf94bf2b6645739c79e9e43"


from openai import OpenAI
import time

import datasets
from datasets import load_dataset
import os
import json
import pandas
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

eval_dataset = load_dataset('parquet', data_files={'eval': 'data/Freeway/DeepSeek/test.parquet'})['eval']

if not os.path.exists("data/sft"):
    os.makedirs("data/sft")
if not os.path.exists("data/sft/eval"):
    os.makedirs("data/sft/eval")
    

for i in range(len(eval_dataset)):
    prompt = eval_dataset[i]["prompt"]
    #print(prompt)
    solution = eval_dataset[i]["solution"]
    available_actions = eval_dataset[i]["available_actions"]
    start_time = time.time()
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=prompt,
        temperature=0.6,
        top_p = 0.95,
        max_tokens=8192
    )
    
    # record (prompt, response, solution, available_actions) to json file
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    print(prompt)
    print(response)
    print(solution)
    print(available_actions)
    
    with open(f"data/sft/eval/{i}.jsonl", "w") as f:
        json.dump({
            "instruction": prompt[1]["content"],
            "system": prompt[0]["content"],
            "response_content": response.choices[0].message.content,
            "response_reasoning_content": response.choices[0].message.reasoning_content,
            "solution": solution,
            "available_actions": available_actions
        }, f)

    # record (prompt, response, solution, available_actions) to pickle file
    
        
