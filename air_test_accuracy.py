import argparse
import datetime 
import pandas as pd
import os
from itertools import cycle
from openai import OpenAI
from transformers import AutoTokenizer
from envs.minatar.environment import Environment
from envs.utils.extract_utils import extract_scratch_pad
from envs.airraid import state_to_description, llm_state_builder, tick
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from generate import generate_vanilla_openai, generate_prompted_s1_openai
from vllm import SamplingParams
def greedy(state_for_llm):
    ships = state_for_llm["reward_states"]
    pos = state_for_llm["player_states"]
    tars = []
    for (x, y, speed, reward) in ships:
        if y <= 0:
            continue
        tars.append((x, y, (y + speed - 1) // speed, reward))
    tars.append((pos, 0, 0, 0))
    tars = sorted(tars, key=lambda x: x[2])
    dp = [0] * 32
    cnt = [0] * 32
    cnt[0] = 1
    pre = [0] * 32
    max_r = 0
    max_cnt = 0
    for i in range(len(tars)):
        for j in range(0, i):
            if cnt[j] == 0:
                continue
            if tars[i][2] - tars[j][2] >= abs(tars[i][0] - tars[j][0]):
                if dp[i] < dp[j] + tars[i][3]:
                    cnt[i] = cnt[j]
                    pre[i] = j
                elif dp[i] == dp[j] + tars[i][3]:
                    cnt[i] += cnt[j]
                dp[i] = max(dp[i], dp[j] + tars[i][3])
        if dp[i] > dp[max_r]:
            max_r = i
            max_cnt = cnt[i]
        elif dp[i] == dp[max_r]:
            max_cnt += cnt[i]
    collect_list = []
    max_reward = dp[max_r]
    while True:
        collect_list.append((tars[max_r][0], tars[max_r][2], tars[max_r][3]))
        if max_r == 0:
            break
        max_r = pre[max_r]
    collect_list.reverse()
    actions = []
    for i in range(len(collect_list) - 1):
        cur, t_cur, _ = collect_list[i]
        nxt, t_nxt, _ = collect_list[i + 1]
        if cur < nxt:
            actions.append('R' * (nxt - cur) + 'S' * (t_nxt - t_cur - nxt + cur))
        elif cur > nxt:
            actions.append('L' * (cur - nxt) + 'S' * (t_nxt - t_cur - cur + nxt))
        else:
            actions.append('S' * (t_nxt - t_cur))
    return actions, max_reward, len(collect_list)

tokenizer = None
def process_entry(i, dic, api_key, args):
    from envs.prompts.ma_airraid_math import LLM_SYSTEM_PROMPT, MATH_PROMPT
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    env = Environment('airraid', sticky_action_prob=0)
    seed = int(dic['seed'])
    env.seed(seed)
    env.reset()
    for _ in range(100):
        env.act(0)
    env.env.pos = (seed * seed) % 9
    answer = dic['actions']
    render = dic['render']
    # print(valid_moves)
    state_for_llm = llm_state_builder(env.env)
    description = state_to_description(state_for_llm)
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
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": MATH_PROMPT + description},
        ],
        sampling_params=sampling_params,
    )
    metrics = []
    text = response['text']
    token_num = response['token_num']
    action = extract_scratch_pad(text, "", valid_actions="LRS")
    tot_reward = 0
    n_state_for_llm = deepcopy(state_for_llm)
    for a in action:
        new_state_for_llm, reward  = tick(n_state_for_llm, a)
        tot_reward += reward
        n_state_for_llm = deepcopy(new_state_for_llm)
    best_actions, max_reward, _ = greedy(n_state_for_llm)
    tot_reward += max_reward
    metrics.append({
        "seed": seed,
        "render": render,
        "best_actions": dic['actions'],
        "planned_reward": tot_reward - max_reward,
        "estimated_reward": tot_reward,
        "max_reward": int(dic['reward']),
        "action": action,
        "response_token_num": token_num,
    })
    if i < 5:
        with open(args.f.replace(".csv", f"_{i}.txt"), 'a') as f:
            f.write(f"{render}\n")
            f.write(f"Action: {action}\n")
            f.write(f"Estimated Reward: {tot_reward}, Planned Reward: {tot_reward - max_reward}, Max Reward: {dic['reward']}\n")
            f.write(f"Response: {text}\n")
            f.write(f"Token Num: {token_num}\n")
    return metrics
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the accuracy of the model.')
    parser.add_argument('--game', type=str, default='freeway', help='Game name')
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--api_keys', nargs='+', type=str, default=["EMPTY"], help='List of API keys for OpenAI')
    parser.add_argument('--base_url', type=str, default=None, help='URL of the model server')
    parser.add_argument('--model', type=str, default = 'deepseek-ai/DeepSeek-R1')
    parser.add_argument('--budget_forcing', type=str, default='no')
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
        if not os.path.exists(f"logs-0604/{game}-acc/{model_name}"):
            os.makedirs(f"logs-0604/{game}-acc/{model_name}")
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.f = f"logs-0604/{game}-acc/{model_name}/{time_stamp}.csv"
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
                executor.submit(process_entry, i, df.iloc[i], next(api_key_cycle), args)
                for i in range(1)
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
    # summarize the metrics
    df = pd.read_csv(args.f)
    optimal_count = 0
    for i in range(len(df)):
        if df['estimated_reward'][i] == df['max_reward'][i]:
            optimal_count += 1
 
    with open(args.f.replace(".csv", ".log"), 'a') as f:
        f.write(f"Total: {len(df)}, Optimal: {optimal_count}\n")
        f.write(f"Optimal Rate: {optimal_count / len(df):.2f}\n")
        f.write(f"Average Response Token Num: {df['response_token_num'].mean():.2f}\n")
        f.write(f"Average Estimated Reward: {df['estimated_reward'].mean():.2f}\n")
        f.write(f"Average Planned Reward: {df['planned_reward'].mean():.2f}\n")
        f.write(f"Average Max Reward: {df['max_reward'].mean():.2f}\n")