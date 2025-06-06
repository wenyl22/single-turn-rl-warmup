import argparse
import datetime 
import pandas as pd
import os
from itertools import cycle
from openai import OpenAI
from transformers import AutoTokenizer
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
from vllm import SamplingParams
from generate import generate_vanilla_openai, generate_prompted_s1_openai
import importlib

tokenizer = None
def process_entry(i, dic, api_key, args, module):
    pre_process = getattr(module, 'pre_process')
    post_process_response = getattr(module, 'post_process_response')
    messages, env = pre_process(dic)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens= args.max_new_tokens,
    )
    if args.budget_forcing == "no":
        generate_func = generate_vanilla_openai
    elif args.budget_forcing == "ps":
        generate_func = generate_prompted_s1_openai
    response = generate_func(
        llm = client,
        tokenizer = tokenizer,
        model = args.model,
        messages=messages,
        sampling_params=sampling_params,
    )
    metrics = post_process_response(i, response, dic, args, env)
    return metrics
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the accuracy of the model.')
    parser.add_argument('--game', type=str, default='freeway', help='Game name')
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--api_keys', nargs='+', type=str, default=["EMPTY"], help='List of API keys for OpenAI')
    parser.add_argument('--base_url', type=str, default=None, help='URL of the model server')
    parser.add_argument('--generation', type=int, default=1, help='Generation number')
    parser.add_argument('--model', type=str, default = 'deepseek-ai/DeepSeek-R1')
    parser.add_argument('--f', type=str, default=None)
    parser.add_argument('--budget_forcing', type=str, default='no', choices=['no', 'ps'], help='Budget forcing method')
    args = parser.parse_args()
    module_name = f"envs.critical.{args.game}"
    module = importlib.import_module(module_name)
    summarize = getattr(module, 'summarize')

    
    if args.f is None:
        game = args.game
        model = args.model
        model_name = args.model.split("/")[-1]
        max_new_tokens = args.max_new_tokens
        if args.model == "deepseek-reasoner":
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
        elif args.model == "deepseek-chat":
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
        if not os.path.exists(f"logs-0605/{game}-acc/{model_name}"):
            os.makedirs(f"logs-0605/{game}-acc/{model_name}")
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.f = f"logs-0605/{game}-acc/{model_name}/{time_stamp}.csv"
        log_file = args.f.replace(".csv", ".log")

        with open(log_file, 'a') as f:
            f.write("Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("\n")
        dataset = f'data/{game}/dataset.csv'
        df = pd.read_csv(dataset)
        
        max_workers = len(args.api_keys)
        api_key_cycle = cycle(args.api_keys)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_entry, i, df.iloc[i], next(api_key_cycle), args, module)
                for i in range(len(df))
            ]
            results = []
            start_time = time.time()
            total = len(futures)
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error: {e}")
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (total - idx) * avg_time / max_workers
                print(f"Completion {idx}/{total}, Avg time {avg_time:.2f}s, ETA: {remaining/60:.2f} mins")
    # dump "results" to a csv file
        all_metrics = []
        for result in results:
            all_metrics.extend(result)
        df = pd.DataFrame(all_metrics)
        df.to_csv(args.f, index=False)
    df = pd.read_csv(args.f)
    summarize(df, args)
