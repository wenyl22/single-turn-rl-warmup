import argparse
import datetime 
import queue
import threading
import time
import torch
import gc
import os
import numpy as np
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark with a specific model.')
    parser.add_argument('--game', type=str, default='freeway', help='Game name')
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--token_per_tick', type=int, default=8192)
    parser.add_argument('--budget-forcing', type=str, default='no', choices=['no', 'prompted', 's1', 'ps', 'br', 'si'], help='budget forcing method')
    parser.add_argument("--method", type=str, default='sa', choices=['hsa', 'lsa', 'pma'], help='framework to use')
    parser.add_argument('--seed_num', type=int, default=8, help='number of seeds to run')
    parser.add_argument('--api_keys', nargs='+', type=str, default=[], help='List of API keys for OpenAI')
    parser.add_argument('--base_url', type=str, default=None, help='URL of the model server')
    parser.add_argument('--model', type=str, default = '')
    parser.add_argument('--low_model', type=str, default = '')
    parser.add_argument('--low_base_url', type=str, default=None, help='URL of the low model server')

    args = parser.parse_args()
    game = args.game
    model = args.model if args.model != "" else args.low_model
    model_name = model.split("/")[-1]
    # model_name = model_name_from_path(model)
    max_new_tokens = args.max_new_tokens
    token_per_tick = args.token_per_tick
    logs = f"logs-0601"
    if not os.path.exists(f"{logs}/{game}_{args.method}/{model_name}"):
        os.makedirs(f"{logs}/{game}_{args.method}/{model_name}")
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file = f"{logs}/{game}_{args.method}/{model_name}/benchmarking_{args.method}_{time_stamp}_{token_per_tick}.log"
    SEEDS = range(0, args.seed_num)

    with open(log_file, 'a') as f:
        f.write("Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
        
    if game == "freeway":
        from envs.freeway import setup_thread_VLLM_client, get_thread_VLLM_client, game_loop as game_func
    elif game == "airraid":
        from envs.airraid import setup_thread_VLLM_client, get_thread_VLLM_client, game_loop as game_func
    elif game == "snake":
        from envs.snake import game_loop as game_func, setup_thread_VLLM_client, get_thread_VLLM_client
    elif game == "pvz":
        from envs.pvz import game_loop as game_func, setup_thread_VLLM_client, get_thread_VLLM_client

    setup_thread_VLLM_client(token_per_tick, args)
    client = get_thread_VLLM_client()

    results = []
    max_workers = len(args.api_keys)
    api_cycle = cycle(range(max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        s_log_file = f"{logs}/{game}_{args.method}/{model_name}/{time_stamp}"
        futures = [
            executor.submit(
                game_func, f"{s_log_file}_{i}.csv" , SEEDS[i], args, next(api_cycle)
            )
            for i in range(args.seed_num)
        ]
        results = []
        total = len(futures)
        for idx, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            with open(log_file, 'a') as f:
                for key, value in result.items():
                    f.write(f"{key}: {value} ")
                f.write("\n---------------------------------------\n")            
    with open(f"{log_file}", 'a') as f:
        for key in results[0].keys():
            if key == "seed":
                continue
            f.write(f"Mean {key}: {np.mean([r[key] for r in results])}\n")
    # kill_vllm_models()