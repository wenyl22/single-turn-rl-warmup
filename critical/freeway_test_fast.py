import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import datetime 
import pandas as pd
from itertools import cycle
from openai import OpenAI
from envs.minatar.environment import Environment
from envs.freeway import state_to_description, llm_state_builder
from envs.prompts.freeway import LLM_SYSTEM_PROMPT, FAST_AGENT_ACTION_PROMPT, FAST_AGENT_CONCLUSION_PROMPT
from utils.extract_utils import extract_boxed
from generate import generate
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
import time
from vllm import SamplingParams

tokenizer = None
def process_entry(i, dic, api_key, args):
    seed = ast.literal_eval(dic['seed'])
    safe_actions = ast.literal_eval(dic['safe_actions'])
    optimal_actions = ast.literal_eval(dic['optimal_actions'])
    env = Environment('freeway', sticky_action_prob=0)
    env.seed(seed[0])
    env.reset()
    for _ in range(seed[1]):
        env.act(0)
    env.env.pos = 9 - seed[2]
    state_for_llm = llm_state_builder(env.env)
    if args.h == "correct":
        scratch_pad = f"**Guidance from a Previous Thinking Model:** Turn \(t_1 = {env.env.game_turn}\) \n"
        scratch_pad += '\n'.join(f"Turn {_ + env.env.game_turn}: {seed[-1][_]}" for _ in range(len(seed[-1])))
        description = state_to_description(state_for_llm, scratch_pad)
    elif args.h == "wrong":
        pass
    else:
        description = state_to_description(state_for_llm)
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    messages = [
        # {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": FAST_AGENT_ACTION_PROMPT + description}
    ]
    sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=8192)
    response = generate(client, args.model, messages, sampling_params)
    text = response['text']
    token_num = response['token_num']
    action = extract_boxed(text)
    safe = action in safe_actions
    optimal = action in optimal_actions   
    metrics = []
    metrics.append({
        'seed': seed,
        'render': dic['render'],
        'need_planning': dic['need_planning'],
        'action': action,
        'safe': safe,
        'optimal': optimal,
        'response_token_num': token_num
    })
    print(seed, safe_actions, optimal_actions, action)
    if i < 5:
        with open(args.f.replace(".csv", f"_{i}.txt"), 'a') as f:
            f.write(f"Seed: {seed}, Need Planning: {dic['need_planning']}\n")
            f.write(f"Safe Actions: {safe_actions}\n")
            f.write(f"Optimal Actions: {optimal_actions}\n")
            f.write(f"LLM Select Action: {action}, Safe: {safe}, Optimal: {optimal}\n")
            f.write(f"Response Token Num: {token_num}\n")
            f.write("\n" + env.env.state_string() + "\n")
            f.write(f"State Description: {description}\n")
            f.write(f"Response: {text}\n")
            f.write("--------------------------------\n")
    return metrics
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the accuracy of the model.')
    parser.add_argument('--game', type=str, default='freeway', help='Game name')
    parser.add_argument('--api_keys', nargs='+', type=str, default=["EMPTY"], help='List of API keys for OpenAI')
    parser.add_argument('--base_url', type=str, default=None, help='URL of the model server')
    parser.add_argument('--model', type=str, default = 'deepseek-ai/DeepSeek-R1')
    parser.add_argument('--f', type=str, default=None)
    parser.add_argument('--h', type=str, default='correct', choices=['correct', 'wrong', 'no'], help='whether there is a help')
    parser.add_argument('--format', type=str, default='action', choices=['action', 'conclusion'], help='format of the prompt')
    args = parser.parse_args()
    if args.f is None:
        game = args.game
        model = args.model
        model_name = args.model.split("/")[-1]
        log_dir = f"logs-0616/{game}-acc/{model_name}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.f = f"{log_dir}/{time_stamp}.csv"
        log_file = args.f.replace(".csv", ".log")

        with open(log_file, 'a') as f:
            f.write("Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
            f.write("\n")
        dataset = f'data/{game}/SingleStepDataset.csv'
        df = pd.read_csv(dataset)
        
        max_workers = len(args.api_keys)
        api_key_cycle = cycle(args.api_keys)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_entry, i, df.iloc[i], next(api_key_cycle), args)
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
        all_metrics = []
        for result in results:
            all_metrics.extend(result)
        df = pd.DataFrame(all_metrics)
        df.to_csv(args.f, index=False)
    df = pd.read_csv(args.f)
    log_file = args.f.replace(".csv", ".log")
    with open(log_file, 'a') as f:
        f.write("Results:\n")
        # aggregate results: mean safe, optimal, response_token_num
        f.write(f"Safe: {df['safe'].mean()}, Optimal: {df['optimal'].mean()}, Response Token Num: {df['response_token_num'].mean()}\n")
        f.write("\nDetailed Results:\n")
        # aggregate by need_planning or not
        for need_planning in df['need_planning'].unique():
            sub_df = df[df['need_planning'] == need_planning]
            if sub_df.empty:
                continue
            f.write(f"Need Planning: {need_planning}\n")
            f.write(f"Safe: {sub_df['safe'].mean()}, Optimal: {sub_df['optimal'].mean()}, Response Token Num: {sub_df['response_token_num'].mean()}\n")
            f.write("\n")