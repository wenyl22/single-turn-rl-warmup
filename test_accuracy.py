import argparse
import datetime 
import pandas as pd
import os
from itertools import cycle
from openai import OpenAI
from transformers import AutoTokenizer
from envs.prompts.ma_freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT
from envs.minatar.environment import Environment
from envs.utils.extract_utils import extract_scratch_pad
from envs.freeway import state_to_description, llm_state_builder, supervise_collision
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
tokenizer = None
def process_entry(i, seed, api_key, args):
    seed = ast.literal_eval(seed)
    env = Environment('freeway', sticky_action_prob=0)
    env.seed(seed[0])
    env.reset()
    for _ in range(seed[2]):
        env.act(0)
    env.env.pos = seed[1]
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    state_for_llm = llm_state_builder(env.env)
    description0 = state_to_description(state_for_llm, need_action=False)
    while True:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages = [
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": LLM_BASE_PROMPT + description0 },
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens = args.max_new_tokens,
                n = args.generation # number of responses to generate
            )
            break
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            time.sleep(1)
    metrics = []
    for r in response.choices:
        text = ""
        if hasattr(r.message, "reasoning_content") and r.message.reasoning_content != None:
            text += "<think>" + r.message.reasoning_content + "</think>\n"
        if r.message.content != None:
            text += r.message.content + "\n"
        token_num = len(tokenizer(text)["input_ids"])
        scratch_pad = extract_scratch_pad(text, "")
        safe = not supervise_collision(state_for_llm, scratch_pad, future_step = len(scratch_pad))
        end_pos = env.env.pos + sum([1 if c == "D" else -1 if c == "U" else 0 for c in scratch_pad])
        if not safe:
            end_pos = 9
        optimal = (safe and end_pos == 0 and len(scratch_pad) == len(seed[5]))
        metrics.append({
            "seed": seed,
            "safe": safe,
            "end_pos": end_pos,
            "optimal": optimal,
            "scratch_pad": scratch_pad,
            "response_token_num": token_num,
        })
        if i < 8:
            with open(args.f.replace(".csv", f"_{i}.txt"), 'a') as f:
                f.write(f"Seed: {seed}, Scratch Pad: {scratch_pad}, Safe: {safe}, End Pos: {end_pos}, Optimal: {optimal}, Response Token Num: {token_num}\n")
                f.write(f"Response: {text}\n")
                f.write("--------------------------------\n")
            
    return metrics
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the accuracy of the model.')
    parser.add_argument('--game', type=str, default='freeway', help='Game name')
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--api_keys', nargs='+', type=str, default=[], help='List of API keys for OpenAI')
    parser.add_argument('--base_url', type=str, default=None, help='URL of the model server')
    parser.add_argument('--generation', type=int, default=1, help='Generation number')
    parser.add_argument('--model', type=str, default = 'deepseek-ai/DeepSeek-R1')
    parser.add_argument('--f', type=str, default=None)
    args = parser.parse_args()
    
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
        if not os.path.exists(f"logs/{game}-acc"):
            os.makedirs(f"logs/{game}-acc")
        if not os.path.exists(f"logs/{game}-acc/{model_name}"):
            os.makedirs(f"logs/{game}-acc/{model_name}")
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.f = f"logs/{game}-acc/{model_name}/{time_stamp}.csv"
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
                executor.submit(process_entry, i, df['seed'][i], next(api_key_cycle), args)
                for i in range(8)
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
    log_file = args.f.replace(".csv", ".log")
    with open(log_file, 'a') as f:
        f.write("Results:\n")
        f.write(f"Total: {len(df)}\n")
        f.write(f"Safe: {df['safe'].sum()}/{len(df)} = {df['safe'].sum()/len(df)}\n")
        f.write(f"Optimal: {df['optimal'].sum()}/{len(df)} = {df['optimal'].sum()/len(df)}\n")
        f.write(f"End Pos: {df['end_pos'].sum()}/{len(df)} = {df['end_pos'].sum()/len(df)}\n")
        f.write(f"Response Token Num: {df['response_token_num'].sum()}/{len(df)} = {df['response_token_num'].sum()/len(df)}\n")
        f.write("\n")
    # aggregate the results, and calculate the mean of safe, optimal, end_pos and response_token_num
    seed_groups = {}
    for i, seed in enumerate(df['seed']):
        seed = ast.literal_eval(seed)
        l = len(seed[5])
        if l not in seed_groups.keys():
            seed_groups[l] = []
        seed_groups[l].append({
            "safe": df['safe'][i],
            "optimal": df['optimal'][i],
            "end_pos": df['end_pos'][i],
            "response_token_num": df['response_token_num'][i],
        })
    with open(log_file, 'a') as f:
        f.write("Results by Optimal Path Length:\n")
        for seed, metrics in seed_groups.items():
            safe = sum([m['safe'] for m in metrics]) / len(metrics)
            optimal = sum([m['optimal'] for m in metrics]) / len(metrics)
            end_pos = sum([m['end_pos'] for m in metrics]) / len(metrics)
            response_token_num = sum([m['response_token_num'] for m in metrics]) / len(metrics)
            f.write(f"Seed: {seed}, Safe: {safe}, Optimal: {optimal}, End Pos: {end_pos}, Response Token Num: {response_token_num}\n")
    # aggregate the results by (seed[3], seed[4])
    seed_groups = {}
    for i, seed in enumerate(df['seed']):
        seed = ast.literal_eval(seed)
        if (seed[3], seed[4]) not in seed_groups:
            seed_groups[(seed[3], seed[4])] = []
        seed_groups[(seed[3], seed[4])].append({
            "safe": df['safe'][i],
            "optimal": df['optimal'][i],
            "end_pos": df['end_pos'][i],
            "response_token_num": df['response_token_num'][i],
        })
    with open(log_file, 'a') as f:
        f.write("Results by Look Forward Step:\n")
        for seed, metrics in seed_groups.items():
            safe = sum([m['safe'] for m in metrics]) / len(metrics)
            optimal = sum([m['optimal'] for m in metrics]) / len(metrics)
            end_pos = sum([m['end_pos'] for m in metrics]) / len(metrics)
            response_token_num = sum([m['response_token_num'] for m in metrics]) / len(metrics)
            f.write(f"Seed: {seed}, Safe: {safe}, Optimal: {optimal}, End Pos: {end_pos}, Response Token Num: {response_token_num}\n")
