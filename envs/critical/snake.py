from envs.prompts.ma_snake_game import LLM_SYSTEM_PROMPT,  GAME_PROMPT
from envs.minatar.environment import Environment
from envs.utils.extract_utils import extract_scratch_pad
import ast
from envs.snake import state_to_description, llm_state_builder, tick
from copy import deepcopy
from snake_ds_generation import bfs
def pre_process(dic):
    env = Environment('snake', sticky_action_prob=0)
    env.env.reset(initialize_food = False)
    env.env.snake = ast.literal_eval(dic['snake'])
    for (x, y, life_span, food_value) in ast.literal_eval(dic['foods']):
        env.env.spawn_food((x, y, life_span, food_value))
    env.env.dir = int(dic['dir'])
    messages = [
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": GAME_PROMPT + state_to_description(llm_state_builder(env.env))},
    ]
    return messages, env
 
def post_process_response(i, response, dic, args, env):
    metrics = []
    text = response["text"]
    token_num = response["token_num"]
    state_for_llm = llm_state_builder(env.env)
    action = extract_scratch_pad(text, "", valid_actions="LRDU")
    if len(action) > 8:
        action = action[:8]  # Limit to 8 characters
    tot_reward = 0
    n_state_for_llm = deepcopy(state_for_llm)
    for a in action:
        new_state_for_llm, reward  = tick(n_state_for_llm, a)
        tot_reward += reward
        n_state_for_llm = deepcopy(new_state_for_llm)
        if n_state_for_llm is None:
            break
    safe = not (n_state_for_llm is None)
    if safe:
        best_actions, max_reward = bfs(n_state_for_llm, step = 8 - len(action))
        if max_reward < -10:
            safe = False
        tot_reward += max_reward
    metrics.append({
        "render": dic['render'],
        "snake": dic['snake'],
        "dir": dic['dir'],
        "foods": dic['foods'],
        "action_label": dic['action_label'],
        "best_actions": dic['best_actions'],
        "safe": safe,
        "reward": tot_reward,
        "max_reward": int(dic['max_reward']),
        "action": action,
        "response_token_num": token_num,
    })
    if i < 5:
        with open(args.f.replace(".csv", f"_{i}.txt"), 'a') as f:
            f.write(f"{dic['render']}\n")
            f.write(f"Action: {action}\n")
            f.write(f"Safe: {safe}, Reward: {tot_reward}, Max Reward: {int(dic['max_reward'])}\n")
            f.write(f"Response: {text}\n")
            f.write(f"Token Num: {token_num}\n")
    return metrics

def summarize(df, args):
    safe_count = 0
    optimal_count = 0
    for i in range(len(df)):
        if df['safe'][i] == True:
            safe_count += 1
            if df['reward'][i] == df['max_reward'][i]:
                optimal_count += 1
    with open(args.f.replace(".csv", ".log"), 'a') as f:
        f.write(f"Total: {len(df)}, Safe: {safe_count}, Optimal: {optimal_count}\n")
        f.write(f"Safe Rate: {safe_count / len(df):.2f}, Optimal Rate: {optimal_count / len(df):.2f}\n")
        f.write(f"Average Response Token Num: {df['response_token_num'].mean():.2f}\n")
    
