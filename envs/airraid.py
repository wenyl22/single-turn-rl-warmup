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
seed_mapping = {0: (60, 465), 1: (183, 460), 2: (245, 444), 3: (592, 440), 4: (696, 476), 5: (794, 513), 6: (945, 478), 7: (987, 424)}
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
    for (x, y, speed, reward) in env.space_ships:
        if y > 0:
            reward_states.append((x, y, speed, reward))
    state_for_llm = {
        'player_states': player_states,
        'reward_states': reward_states,
    }
    return state_for_llm

def state_to_description(state_for_llm, state_prediction = 0, scratch_pad = None):
    description = ""
    if state_prediction == 0:
        description += f"### **Game State**\n**Current Turn Player Position:** \(({state_for_llm['player_states']}, 0)\) \n"
        if scratch_pad is not None:
            description += f"**Plan Advice**: {','.join(scratch_pad)}\n"
        description += f"**Current Turn Reward State**:\n"
    else:
        description += f"**Predicted Reward State After {state_prediction} Turns**:\n"
    description += REWARD_STATE
    for (x, y, speed, reward) in state_for_llm['reward_states']:
        description += f"| {reward} | \({x}, {y}\) | {speed} |\n"
    return description
