import argparse
import datetime 
import queue
import threading
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import gc
import os
import numpy as np
from openai import OpenAI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark with a specific model.')
    parser.add_argument('--game', type=str, default='freeway', help='Game name')
    parser.add_argument('--parallel_size', default=8, type=int, help='number of parallel envs to run')
    parser.add_argument('--max_num_seqs', default=8, type=int, help='number of parallel threads to run')
    parser.add_argument('--tensor_parallel_size', default=4, type=int, help="tensor parallel size to load model with vllm")
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


    if not os.path.exists(f"logs/{game}_{args.method}/{model_name}"):
        os.makedirs(f"logs/{game}_{args.method}/{model_name}")
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")      
    log_file = f"logs/{game}_{args.method}/{model_name}/benchmarking_{args.method}_{time_stamp}_{token_per_tick}.log"
    SEEDS = range(0, args.seed_num)

    with open(log_file, 'a') as f:
        f.write("Arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n")
        
    if args.budget_forcing == "no":
        from generate import generate_vanilla as generate_func
    elif args.budget_forcing == "prompted":
        from generate import generate_prompted as generate_func
    elif args.budget_forcing == "s1":
        from generate import generate_s1 as generate_func
    elif args.budget_forcing == "ps":
        from generate import generate_prompted_s1 as generate_func

    if game == "freeway":
        from envs.freeway import setup_thread_VLLM_client, get_thread_VLLM_client
        from envs.freeway import pma_freeway_game_loop as game_func
    elif game == "overcooked":
        from envs.overcooked import overcooked_game_loop as game_func, setup_thread_VLLM_client, get_thread_VLLM_client
    elif game == "asterix":
        from envs.asterix import asterix_game_loop as game_func, setup_thread_VLLM_client, get_thread_VLLM_client

    setup_thread_VLLM_client(token_per_tick, args)
    client = get_thread_VLLM_client()

    if args.api_keys == []:
        llm = LLM(model, gpu_memory_utilization=0.95, tensor_parallel_size=args.tensor_parallel_size, max_num_seqs=args.max_num_seqs, disable_custom_all_reduce=True, max_model_len=16384)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = None
    results = []
    for i in range(0, len(SEEDS), args.parallel_size):
        batch = SEEDS[i: i + args.parallel_size]
        threads = []
        return_queue = queue.Queue()  

        def thread_target(*args):
            result = game_func(*args)
            return_queue.put(result)
        for s in batch:
            s_log_file = f"logs/{game}_{args.method}/{model_name}/{time_stamp}_{max_new_tokens}_{s}.csv"
            thread = threading.Thread(target=thread_target, args=(s_log_file, s, args))
            threads.append(thread)
            thread.start()
            time.sleep(0.1)
        num_alive_threads = len(threads)
        queries = []
        while num_alive_threads > 0:
            for k in client.query_queues.keys():
                try:
                    query, tp = client.query_queues[k].get_nowait()
                    queries.append((k, query, tp))
                except queue.Empty:
                    pass            
            num_alive_threads = 0
            for thread in threads:
                if thread.is_alive():
                    num_alive_threads += 1
            if len(queries) < num_alive_threads:
                continue
            if num_alive_threads == 0:
                break
            sampling_params = queries[0][2]
            outputs = generate_func(llm, tokenizer, [message for _, message, __ in queries if len(message) != 0], sampling_params)

            responses = []
            index = 0 
            for output in outputs:
                while index < len(queries) and len(queries[index][1]) == 0:
                    index += 1
                    responses.append(dict(text="", token_num=[]))
                index += 1
                text = output[0]
                token_num = output[1]
                responses.append(dict(text=text, token_num=token_num))
            while index < len(queries):
                assert(len(queries[index][1]) == 0)
                index += 1
                responses.append(dict(text="", token_num=0))
            for (k, query, tp), response in zip(queries, responses):
                client.response_queues[k].put_nowait(response)
            queries = []

            time.sleep(0.05)
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
        
        for thread in threads:
            thread.join()
            
        while not return_queue.empty():
            ret = return_queue.get()
            results.append(ret)
            with open(f"{log_file}", 'a') as f:
                for key, value in ret.items():
                    f.write(f"{key}: {value} ")
                f.write("\n---------------------------------------\n")
    with open(f"{log_file}", 'a') as f:
        for key in results[0].keys():
            if key == "seed":
                continue
            f.write(f"Mean {key}: {np.mean([r[key] for r in results])}\n")
    # kill_vllm_models()