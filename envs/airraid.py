import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time 
import pandas as pd
from minatar.environment import Environment
from minatar.environments.airraid import Env
from envs.utils.extract_utils import extract_scratch_pad, extract_boxed
from envs.utils.client_utils import ApiThreadedLLMClient
from vllm import SamplingParams
from envs.prompts.ma_airraid_math import LLM_SYSTEM_PROMPT, MATH_PROMPT, MATH_PROMPT_LOW_LEVEL, REWARD_STATE

VLLM_client = None 
seed_mapping = {0: (39, 418), 1: (78, 474), 2: (89, 448), 3: (97, 474), 4: (116, 472), 5: (321, 429), 6: (404, 488), 7: (551, 356)}
def setup_thread_VLLM_client(token_per_tick, args):
    global VLLM_client
    VLLM_client = ApiThreadedLLMClient(token_per_tick, args) 
   
def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client
        
def game_loop(log_file, seed, args, thread_id):

    client = VLLM_client
    client.add_new_thread(thread_id)
    env = Environment('airraid', sticky_action_prob=0)
    if seed in seed_mapping:
        seed = seed_mapping[seed][0]
        env.seed(seed)
        env.reset()
    else:
        env.seed(seed)
        env.reset()
    reward = 0
    game_turn = 0
    scratch_pad = ""
    start_time = time.time()
    terminal = False
    logs = {'description': [], 'render':[], 'supervisor_response': [], 'plan_agent_response':[], 'scratch_pad': [], 'selected_agent': [], 'selected_action': [], "reward": []}
    while True:
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm)
        logs['description'].append(state_description)
        logs['render'].append('\n' + env.env.state_string())
        log_supervisor_response, log_selected_agent, log_plan_agent_response, log_selected_action = "", "", "", ""
        # ### --- High Level Agent --- ###
        if args.method != "lsa":
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": MATH_PROMPT + state_description}
            ]
            # print(MATH_PROMPT + state_description)
            # exit(0)
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new_tokens - 5)
        # OPTION2: Interrupt the thread with new state.
        if args.budget_forcing == "si":
            pass
            # sampling_params.max_tokens = client.token_per_tick - 5
            # end, log_plan_agent_response = client.run_inference_with_interruption(thread_id, messages, "", sampling_params)
            # if end:
            #     scratch_pad = extract_scratch_pad_lr(log_plan_agent_response, scratch_pad)
        # OPTION1: Automatically drop message if the thread is planning state for previous turns.
        else:
            turns = client.token_queue_len[thread_id] // client.token_per_tick
            # The message will be automatically dropped if the thread is planning state for previous turns.
            log_plan_agent_response = client.run_inference(thread_id, messages, "", sampling_params)
            if log_plan_agent_response != "": # agent responds with a plan
                scratch_pad = extract_scratch_pad(log_plan_agent_response, scratch_pad, valid_actions="LRS")
                scratch_pad = scratch_pad[turns:] if turns < len(scratch_pad) else ""
        logs['plan_agent_response'].append(log_plan_agent_response)
        if scratch_pad == "":
            scratch_pad = "S"
        logs['scratch_pad'].append(scratch_pad)
        ### --- Low Level Agent --- ###
        action = 'S'
        if args.method == "hsa":
            action = scratch_pad[0]
            log_supervisor_response = "Follow Plan"
        else:
            state_description = state_to_description(state_for_llm, 0, scratch_pad)
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": MATH_PROMPT_LOW_LEVEL + state_description}
            ]
            # print(MATH_PROMPT_LOW_LEVEL + state_description)
            # exit(0)
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
            log_supervisor_response = client.run_low_level_inference(thread_id, messages, sampling_params)
            action = extract_boxed(log_supervisor_response)
        log_selected_action = action
        if action == scratch_pad[0]:
            log_selected_agent = "B. Follow Plan Agent"
            scratch_pad = scratch_pad[1:]
        else:
            log_selected_agent = "C. React Agent"
            scratch_pad = ""
        action = 0 if action == 'S' else 1 if action == 'L' else 3
        logs['supervisor_response'].append(log_supervisor_response)
        logs['selected_agent'].append(log_selected_agent)
        logs['selected_action'].append(log_selected_action)
        r, terminal = env.act(action)
        game_turn += 1
        reward += r
        logs["reward"].append(reward)
        df = pd.DataFrame(logs)
        df.to_csv(log_file)
        if game_turn >= 50:
            break
        print(f"Thread {thread_id} - Game Turn: {game_turn}, Reward: {reward}")

    return {
        'seed': seed,
        'game_turn': game_turn,
        'reward': reward,
        'game_time': time.time() - start_time
    }


def llm_state_builder(env: Env):
    player_states = env.pos
    reward_states = []
    # print(env.space_ships)
    for j in range(10):
        for (x, y, speed, reward) in env.space_ships[j]:
            assert y > 0
            reward_states.append((x, y, speed, reward))
    state_for_llm = {
        'player_states': player_states,
        'reward_states': reward_states,
    }
    return state_for_llm

def state_to_description(state_for_llm, state_prediction = 0, scratch_pad = None):
    description = ""
    description += f"### **Game State**\n**Current Turn Player Position:** \(({state_for_llm['player_states']}, 0)\) \n"
    description += f"**Current Turn Reward State**:\n"
    description += REWARD_STATE
    for (x, y, speed, reward) in state_for_llm['reward_states']:
        description += f"| {reward} | \({x}, {y}\) | {speed} |\n"
    if scratch_pad is not None:
        description += f"**Plan Advice**: {",".join(scratch_pad)}\n"
    return description

def tick(state_for_llm, action):
    """
    Simulate one tick of the game with the given action.
    Returns the new state and the reward.
    """
    player_states = state_for_llm['player_states']
    reward_states = state_for_llm['reward_states']
    
    if action == "S":  # Stay
        pass
    elif action == "L":  # Move left
        player_states = max(0, player_states - 1)
    elif action == "R":  # Move right
        player_states = min(9, player_states + 1)
    new_reward_states = []
    ret = 0
    for (x, y, speed, reward) in reward_states:
        new_y = y - speed
        if new_y <= 0:
            if x == player_states:
                ret += reward
        else:
            new_reward_states.append((x, new_y, speed, reward))
    return {'player_states': player_states, 'reward_states': new_reward_states}, ret

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
