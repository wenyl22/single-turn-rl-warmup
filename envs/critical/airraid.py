from envs.minatar.environment import Environment
from envs.utils.extract_utils import extract_scratch_pad
from envs.airraid import state_to_description, llm_state_builder, tick, greedy
from copy import deepcopy
from envs.prompts.ma_airraid_math import LLM_SYSTEM_PROMPT, MATH_PROMPT


def pre_process(dic):
    env = Environment('airraid', sticky_action_prob=0)
    seed = int(dic['seed'])
    env.seed(seed)
    env.reset()
    for _ in range(100):
        env.act(0)
    env.env.pos = (seed * seed) % 9
    description = state_to_description(llm_state_builder(env.env))
    messages=[
        {"role": "system", "content": LLM_SYSTEM_PROMPT},
        {"role": "user", "content": MATH_PROMPT + description},
    ]
    return messages, env

def post_process_response(i, response, dic, args, env):
    metrics = []
    text = response['text']
    token_num = response['token_num']
    action = extract_scratch_pad(text, "", valid_actions="LRS")
    tot_reward = 0
    state_for_llm = llm_state_builder(env.env)
    n_state_for_llm = deepcopy(state_for_llm)
    for a in action:
        new_state_for_llm, reward  = tick(n_state_for_llm, a)
        tot_reward += reward
        n_state_for_llm = deepcopy(new_state_for_llm)
    best_actions, max_reward, _ = greedy(n_state_for_llm)
    tot_reward += max_reward
    metrics.append({
        "seed": dic['seed'],
        "render": dic['render'],
        "best_actions": dic['actions'],
        "planned_reward": tot_reward - max_reward,
        "estimated_reward": tot_reward,
        "max_reward": int(dic['reward']),
        "action": action,
        "response_token_num": token_num,
    })
    if i < 5:
        with open(args.f.replace(".csv", f"_{i}.txt"), 'a') as f:
            f.write(f"{dic['render']}\n")
            f.write(f"Action: {action}\n")
            f.write(f"Estimated Reward: {tot_reward}, Planned Reward: {tot_reward - max_reward}, Max Reward: {dic['reward']}\n")
            f.write(f"Response: {text}\n")
            f.write(f"Token Num: {token_num}\n")
    return metrics

def summarize(df, args):
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