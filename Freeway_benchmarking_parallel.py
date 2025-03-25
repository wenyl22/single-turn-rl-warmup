from __future__ import print_function
import argparse
import datetime 
import queue
import threading
import time
from transformers import AutoTokenizer
from Freeway_agent import setup_thread_VLLM_client, get_thread_VLLM_client
from Freeway_benchmarking import game_loop
from vllm import LLM, SamplingParams
import torch
import gc
import os
import numpy as np

setup_thread_VLLM_client()

parser = argparse.ArgumentParser(description='Run Hanabi benchmark with a specific model.')
parser.add_argument('--model_name', type=str, default = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
parser.add_argument('--parallel_size', default=8, type=int, help='number of parallel env to run')
parser.add_argument('--max_num_seqs', default=16, type=int, help='number of parallel env to run')
parser.add_argument('--max_num_seeds', default=8, type=int, help='number of seeds')
parser.add_argument('--tensor_parallel_size', default=4, type=int, help="tensor parallel size to load model with vllm")
parser.add_argument('--start_seed', default=1000, type=int, help='number of seeds')
parser.add_argument('--max_new_tokens', type=int, default=2000)
parser.add_argument('--token_per_tick', type=int, default=2000)
args = parser.parse_args()


tensor_parallel_size = args.tensor_parallel_size
model_name = args.model_name
max_new_tokens = args.max_new_tokens
token_per_tick = args.token_per_tick
print(f'Benchmarking model: {model_name}')

if not os.path.exists(f"logs/freeway/{model_name.split('/')[-1]}"):
    os.makedirs(f"logs/freeway/{model_name.split('/')[-1]}")
    
log_file = f"logs/freeway/{model_name.split('/')[-1]}/freeway_benchmarking_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

SEEDS = [i+1 for i in range(args.start_seed, args.start_seed + args.max_num_seeds)] 
if __name__ == "__main__":
    llm = LLM(args.model_name, gpu_memory_utilization=0.8, tensor_parallel_size=tensor_parallel_size, max_num_seqs=args.max_num_seqs,
              disable_custom_all_reduce=True)
    client = get_thread_VLLM_client()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    with open(log_file, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Max new tokens: {max_new_tokens}\n")
        f.write(f"Token per tick: {token_per_tick}\n")
        f.write(f"Seeds: {SEEDS}\n")
    results = []

    for i in range(0, len(SEEDS), args.parallel_size):
        batch = SEEDS[i: i + args.parallel_size]
        threads = []
        return_queue = queue.Queue()  

        def thread_target(*args):
            result = game_loop(*args)
            return_queue.put(result)
        for s in batch:
            thread = threading.Thread(target=thread_target, args=(model_name, max_new_tokens, token_per_tick, "thread", s))
            threads.append(thread)
            thread.start()
            time.sleep(0.1)
        
        num_alive_threads = len(threads)
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.6, top_p=0.95, logprobs=True)
        
        queries = []
        while num_alive_threads > 0:
            for k in client.query_queues.keys():
                try:
                    query = client.query_queues[k].get_nowait()
                    queries.append((k, query))
                except queue.Empty:
                    pass
            
            num_alive_threads = 0
            for thread in threads:
                if thread.is_alive():
                    num_alive_threads += 1

            if len(queries) >= num_alive_threads:
                prompts = [prompt for _, prompt in queries]
                outputs = llm.generate(prompts, sampling_params)
                responses = []
                for output in outputs:
                    text = output.outputs[0].text
                    token_ids = output.outputs[0].token_ids
                    responses.append(dict(text=text, token_ids=token_ids))
                for (k, query), response in zip(queries, responses):
                    client.response_queues[k].put_nowait(response)
                queries = []

            time.sleep(0.05)

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
        
        for thread in threads:
            thread.join()
            
        while not return_queue.empty():
            results.append(return_queue.get())
        with open(f"{log_file}", 'a') as f:
            for r in results:
                f.write(f"Seed: {r[0]}, Game turns: {r[1]}, Game time: {r[2]}\n")
            f.write(f"---------------------------------------\n")
    with open(f"{log_file}", 'a') as f:
        f.write(f"Mean Game Turns: {np.mean([r[1] for r in results])}\n")
        f.write(f"Mean Game Time: {np.mean([r[2] for r in results])}\n")
