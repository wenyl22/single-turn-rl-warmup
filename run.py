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
from tqdm import tqdm
from openai import OpenAI
import requests
import json

### Tested ###

def model_name_from_path(model_path):
    return model_path.split("/")[-2] if model_path.endswith("/") else model_path.split("/")[-1]  

def test_simple_inference(port, model_name):
    """Test a simple inference request"""
    url = f"http://localhost:{port}/v1/completions"
    
    payload = {
        "model": model_name,
        "prompt": "Hello, world!",
        "max_tokens": 10,
        "temperature": 0.7
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Inference successful on port {port}")
            print(f"   Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Inference failed on port {port} with status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Inference error on port {port}: {e}")
        return False

def launch_vllm_models(
        main_model_path, 
        extract_model_path,
        main_tp_size = 4,
        extract_tp_size = 4,
        port1=8000,
        port2=8001
    ):
    """
    Launch two vLLM models with the specified configurations.
    
    Args:
        main_model_path (str): Path to the first model
        extract_model_path (str): Path to the second model
        main_tp_size (int): Tensor parallel size for the first model
        extract_tp_size (int): Tensor parallel size for the second model
    """
    
    # check if main_tp_size + extract_tp_size > available_gpus
    available_gpus = torch.cuda.device_count()
    extract_tp_size = min(extract_tp_size, available_gpus - main_tp_size)
    assert extract_tp_size > 0, f"Not enough GPUs available. Required at least: {main_tp_size + 1}, Available: {available_gpus}"

    main_model = model_name_from_path(main_model_path)
    extract_model = model_name_from_path(extract_model_path)
        
    def gpu_list(start, num):
        return ",".join([str(i) for i in range(start, start + num)])

    # Launch the first model and get its PID for later killing
    cmd1 = [
        f"CUDA_VISIBLE_DEVICES='{gpu_list(0, main_tp_size)}'",
        "python -m vllm.entrypoints.openai.api_server",
        f"--model {main_model_path}",
        f"--tensor-parallel-size {main_tp_size}",
        "--host 'localhost'",
        f"--port {port1}",
        "--gpu-memory-utilization 0.9",
        f"--served-model-name {main_model}",
        "--trust-remote-code",
        "--disable-custom-all-reduce",
        "--max-model-len 16384",
        "&"
    ]

    cmd1 = " ".join(cmd1)
    os.system(cmd1)
    print(f"Started main model with command: {cmd1}")

    # Launch the second model and get its PID for later killing
    cmd2 = [
        f"CUDA_VISIBLE_DEVICES='{gpu_list(main_tp_size, extract_tp_size)}'",
        "python -m vllm.entrypoints.openai.api_server",
        f"--model {extract_model_path}",
        f"--tensor-parallel-size {extract_tp_size}",
        "--host 'localhost'",
        f"--port {port2}",
        "--gpu-memory-utilization 0.9",
        f"--served-model-name {extract_model}",
        "--trust-remote-code",
        "--disable-custom-all-reduce",
        "--max-model-len 16384",
        "&"
    ]
    cmd2 = " ".join(cmd2)
    os.system(cmd2)
    print(f"Started extract model with command: {cmd2}")
    
    # Wait for the models to be ready
    while True:
        if test_simple_inference(port=port1, model_name=main_model) and test_simple_inference(port=port2, model_name=extract_model):
            break
        time.sleep(30)
    print("Models are ready.")
    
def kill_vllm_models():
    """
    Kill the vLLM models that were launched.
    """
    os.system("pkill -f 'python -m vllm.entrypoints.openai.api_server'")
    print("Killed all vLLM models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark with a specific model.')
    parser.add_argument('--difficulty', type=int, default=8, help='difficulty level')
    parser.add_argument('--game', type=str, default='freeway', help='Game name')
    parser.add_argument('--model', type=str, default = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--extract_model', type=str, default = '/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/model/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--api_key', type=str, default = None, help='API key')
    parser.add_argument('--parallel_size', default=8, type=int, help='number of parallel envs to run')
    parser.add_argument('--max_num_seqs', default=8, type=int, help='number of parallel threads to run')
    parser.add_argument('--tensor_parallel_size', default=4, type=int, help="tensor parallel size to load model with vllm")
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--token_per_tick', type=int, default=8192)
    parser.add_argument('--budget-forcing', type=str, default='no', choices=['no', 'prompted', 's1', 'ps'], help='budget forcing method')
    parser.add_argument('--ma', default=False, help='use multi-agent or not', action='store_true')
    parser.add_argument('--seed_num', type=int, default=8, help='number of seeds to run')
    args = parser.parse_args()
    game = args.game
    model = args.model
    model_name = model_name_from_path(model)
    max_new_tokens = args.max_new_tokens
    token_per_tick = args.token_per_tick


    if not os.path.exists(f"logs/{game}/{model_name}"):
        os.makedirs(f"logs/{game}/{model_name}")
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")      
    log_file = f"logs/{game}/{model_name}/benchmarking_{args.budget_forcing}_{time_stamp}_{token_per_tick}.log"
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
        if not args.ma:
            from envs.freeway import freeway_game_loop as game_func
        else:
            from envs.freeway import ma_freeway_game_loop as game_func
    elif game == "overcooked":
        from envs.overcooked import overcooked_game_loop as game_func, setup_thread_VLLM_client, get_thread_VLLM_client
    elif game == "asterix":
        from envs.asterix import asterix_game_loop as game_func, setup_thread_VLLM_client, get_thread_VLLM_client

    setup_thread_VLLM_client(token_per_tick)
    main_client, extract_client = get_thread_VLLM_client()

    # if args.api_key is None:
    #     llm = LLM(model, gpu_memory_utilization=0.95, tensor_parallel_size=args.tensor_parallel_size, max_num_seqs=args.max_num_seqs, disable_custom_all_reduce=True, max_model_len=16384)
    #     tokenizer = AutoTokenizer.from_pretrained(args.model)
    # else:
    #     llm = OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")
    #     tokenizer = None
    
    # Launch Model and Extract_Model via VLLM api
    launch_vllm_models(args.model, args.extract_model, main_tp_size=args.tensor_parallel_size)

    main_llm = OpenAI(api_key="", base_url="http://localhost:8000")
    extract_llm = OpenAI(api_key="", base_url="http://localhost:8001")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    extract_tokenizer = AutoTokenizer.from_pretrained(args.extract_model)
    
    results = []
    for i in range(0, len(SEEDS), args.parallel_size):
        batch = SEEDS[i: i + args.parallel_size]
        threads = []
        return_queue = queue.Queue()  

        def thread_target(*args):
            result = game_func(*args)
            return_queue.put(result)
        for s in batch:
            s_log_file = f"logs/{game}/{model_name}/{time_stamp}_{args.budget_forcing}_{token_per_tick}_{s}.csv"
            thread = threading.Thread(target=thread_target, args=(s_log_file, s, args.difficulty))
            threads.append(thread)
            thread.start()
            time.sleep(0.1)
        
        num_alive_threads = len(threads)
        queries = []
        while num_alive_threads > 0:
            for k in main_client.query_queues.keys():
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
            tp = queries[0][2]
            assert tp in ["reasoning", "supervise"]
            if tp == "reasoning": # use args.model, i.e. llm
                sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.6, top_p=0.95)
                outputs = generate_func(llm, tokenizer, [message for _, message, __ in queries if len(message) != 0], sampling_params, max_new_tokens)
            elif tp == "supervise": # use args.model, i.e. llm
                sampling_params = SamplingParams(max_tokens=1024, temperature=0.6, top_p=0.95)
                outputs = generate_func(llm, tokenizer, [message for _, message, __ in queries if len(message) != 0], sampling_params, 512)

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
    kill_vllm_models()