from envs.freeway import state_to_description, llm_state_builder, supervise_collision
from envs.prompts.ma_freeway_math import MATH_PROMPT, LLM_SYSTEM_PROMPT
from envs.minatar.environment import Environment
from envs.freeway import llm_state_builder, state_to_description
from envs.utils.extract_utils import extract_scratch_pad
import ast
def pre_process(dic):
    seed = ast.literal_eval(dic['seed'])
    env = Environment('freeway', sticky_action_prob=0)
    env.seed(seed[0])
    env.reset()
    for _ in range(seed[2]):
        env.act(0)
    env.env.pos = seed[1]
    state_for_llm = llm_state_builder(env.env)
    description = state_to_description(state_for_llm)
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": MATH_PROMPT + description}
    ]
    return messages, env

def post_process_response(i, response, dic, args, env):
    seed = ast.literal_eval(dic['seed'])
    metrics = []
    text = response['text']
    token_num = response['token_num']
    state_for_llm = llm_state_builder(env.env)
    scratch_pad = extract_scratch_pad(text, "", valid_actions="UDS")
    safe = not supervise_collision(state_for_llm, scratch_pad, future_step = len(scratch_pad))
    end_pos = env.env.pos + sum([1 if c == "D" else -1 if c == "U" else 0 for c in scratch_pad])
    if not safe:
        end_pos = 9
    optimal = (safe and end_pos == 0 and len(scratch_pad) == len(seed[5]))
    metrics.append({
        "seed": seed,
        "render": dic['render'],
        "safe": safe,
        "end_pos": end_pos,
        "optimal": optimal,
        "scratch_pad": scratch_pad,
        "response_token_num": token_num,
    })
    if i < 20:
        with open(args.f.replace(".csv", f"_{i}.txt"), 'a') as f:
            f.write(f"Seed: {seed}, Scratch Pad: {scratch_pad}, Safe: {safe}, End Pos: {end_pos}, Optimal: {optimal}, Response Token Num: {token_num}\n")
            f.write(f"Response: {text}\n")
            f.write("--------------------------------\n")
    return metrics

def summarize(df, args):
    for i in range(len(df)):
        messages, env = pre_process(df.iloc[i])
        response ={
            'text': f'\\boxed{{{df.iloc[i]["scratch_pad"]}}}',
            'token_num': len(df.iloc[i]["scratch_pad"])
        }
        metrics = post_process_response(i, response, df.iloc[i], args, env)
        df.at[i, 'safe'] = metrics[0]['safe']
        df.at[i, 'optimal'] = metrics[0]['optimal']
        df.at[i, 'end_pos'] = metrics[0]['end_pos']
        df.at[i, 'response_token_num'] = metrics[0]['response_token_num']
            
    log_file = args.f.replace(".csv", ".log")
    with open(log_file, 'a') as f:
        f.write("Results:\n")
        f.write(f"Total: {len(df)}\n")
        f.write(f"Safe: {df['safe'].sum()}/{len(df)} = {df['safe'].sum()/len(df)}\n")
        f.write(f"Optimal: {df['optimal'].sum()}/{len(df)} = {df['optimal'].sum()/len(df)}\n")
        f.write(f"End Pos: {df['end_pos'].sum()}/{len(df)} = {df['end_pos'].sum()/len(df)}\n")
        f.write(f"Response Token Num: {df['response_token_num'].sum()}/{len(df)} = {df['response_token_num'].sum()/len(df)}\n")
        f.write("\n")
    seed_groups = {}
    for i, seed in enumerate(df['seed']):
        seed = ast.literal_eval(seed)
        l = len(seed[5])
        mapping = {6: 6, 7: 6, 8: 6, 9: 6, 10: 10, 11: 10, 12: 10, 13: 10, 14: 14, 15: 14, 16: 14, 17: 14}
        l = mapping[l]
        if l not in seed_groups.keys():
            seed_groups[l] = []
        seed_groups[l].append({
            "safe": df['safe'][i],
            "optimal": df['optimal'][i],
            "end_pos": df['end_pos'][i],
            "response_token_num": df['response_token_num'][i],
        })
    # sort the seed_groups by key
    seed_groups = dict(sorted(seed_groups.items(), key=lambda x: x[0]))
    with open(log_file, 'a') as f:
        f.write("Results by Optimal Path Length:\n")
        for seed, metrics in sorted(seed_groups.items()):
            safe = sum([m['safe'] for m in metrics]) / len(metrics)
            optimal = sum([m['optimal'] for m in metrics]) / len(metrics)
            end_pos = sum([m['end_pos'] for m in metrics]) / len(metrics)
            response_token_num = sum([m['response_token_num'] for m in metrics]) / len(metrics)
            f.write(f"Seed: {seed}, Safe: {safe}, Optimal: {optimal}, End Pos: {end_pos}, Response Token Num: {response_token_num}\n")

    # aggregate the results by (seed[3], seed[4])
    seed_groups = {}
    for i, seed in enumerate(df['seed']):
        seed = ast.literal_eval(seed)
        if seed[4] != 0:
            continue
        if seed[3] not in seed_groups:
            seed_groups[seed[3]] = []
        seed_groups[seed[3]].append({
            "safe": df['safe'][i],
            "optimal": df['optimal'][i],
            "end_pos": df['end_pos'][i],
            "response_token_num": df['response_token_num'][i],
        })
    seed_groups = dict(sorted(seed_groups.items(), key=lambda x: x[0]))
    with open(log_file, 'a') as f:
        f.write("Results by Look Forward Step:\n")
        for seed, metrics in seed_groups.items():
            safe = sum([m['safe'] for m in metrics]) / len(metrics)
            optimal = sum([m['optimal'] for m in metrics]) / len(metrics)
            end_pos = sum([m['end_pos'] for m in metrics]) / len(metrics)
            response_token_num = sum([m['response_token_num'] for m in metrics]) / len(metrics)
            f.write(f"Seed: {seed}, Safe: {safe}, Optimal: {optimal}, End Pos: {end_pos}, Response Token Num: {response_token_num}\n")
