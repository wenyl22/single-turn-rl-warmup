import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time 
import pandas as pd
from minatar.environment import Environment
from minatar.environments.pvz import Env
from envs.utils.extract_utils import extract_boxed
from envs.utils.client_utils import ApiThreadedLLMClient
from vllm import SamplingParams
from envs.prompts.ma_pvz_game import LLM_SYSTEM_PROMPT, GAME_PROMPT, GAME_PROMPT_LOW_LEVEL

VLLM_client = None 
seed_mapping = {0: 1000, 1: 1001, 2: 1002, 3: 1003, 4: 1004, 5: 1005, 6: 1006, 7: 1007}
def setup_thread_VLLM_client(token_per_tick, args):
    global VLLM_client
    VLLM_client = ApiThreadedLLMClient(token_per_tick, args) 
   
def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client
        
def game_loop(log_file, seed, args, thread_id):
    client = VLLM_client
    client.add_new_thread(thread_id)
    env = Environment('pvz', sticky_action_prob=0)
    if seed in seed_mapping:
        seed = seed_mapping[seed]
        env.seed(seed)
        env.reset()
    else:
        env.seed(seed)
        env.reset()
    reward = 0
    game_turn = 0
    scratch_pad = []
    start_time = time.time()
    logs = {'description': [], 'render':[], 'supervisor_response': [], 'plan_agent_response':[], 'scratch_pad': [], 'selected_agent': [], 'selected_action': []}
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
                {"role": "user", "content": GAME_PROMPT + state_description}
            ]
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new_tokens - 5)
        if args.budget_forcing == "si":
            pass
        else:
            turns = client.token_queue_len[thread_id] // client.token_per_tick
            # The message will be automatically dropped if the thread is planning state for previous turns.
            log_plan_agent_response = client.run_inference(thread_id, messages, "", sampling_params)
            if log_plan_agent_response != "": # agent responds with a plan
                scratch_pad_txt = extract_boxed(log_plan_agent_response, default_value="None")
                # Each turn takes up one line, begins by: Turn x: .....
                # Formalize them in a list: scratch_pad = [..., ...]
                scratch_pad = []
                for act in scratch_pad_txt.split('\n'):
                    scratch_pad.append(act.split(':')[-1].strip())       
                scratch_pad = scratch_pad[turns:] if turns < len(scratch_pad) else []
        logs['plan_agent_response'].append(log_plan_agent_response)
        if scratch_pad == []:
            scratch_pad = ["None"]
        logs['scratch_pad'].append(scratch_pad)
        ### --- Low Level Agent --- ###
        action = "None"
        if args.method == "hsa":
            action = scratch_pad[0]
            log_supervisor_response = "Follow Plan"
        else:
            state_description = state_to_description(state_for_llm, 0, scratch_pad)
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": GAME_PROMPT_LOW_LEVEL + state_description}
            ]
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
            log_supervisor_response = client.run_low_level_inference(thread_id, messages, sampling_params)
            action = extract_boxed(log_supervisor_response, default_value="None")
        log_selected_action = action
        if action == scratch_pad[0]:
            log_selected_agent = "B. Follow Plan Agent"
            scratch_pad = scratch_pad[1:]
        else:
            log_selected_agent = "C. React Agent"
            scratch_pad = []
        logs['supervisor_response'].append(log_supervisor_response)
        logs['selected_agent'].append(log_selected_agent)
        logs['selected_action'].append(log_selected_action)
        df = pd.DataFrame(logs)
        df.to_csv(log_file)
        r , terminal = env.act(action)
        if terminal:
            print(f"Thread {thread_id} - Game Over at Turn: {game_turn}, Total Reward: {reward}")
            break
        game_turn += 1
        reward += r
        print(f"Thread {thread_id} - Game Turn: {game_turn}, Reward: {reward}")

    return {
        'seed': seed,
        'game_turn': game_turn,
        'reward': reward,
        'game_time': time.time() - start_time
    }


def llm_state_builder(env: Env):
    return {
        "map": env.state_string(), 
        "turn": env.board['total_rounds'],
    }

def state_to_description(state_for_llm, state_prediction = 0, scratch_pad = None):
    description = """## Current game state\n"""
    if scratch_pad is not None and scratch_pad != []:
        try:
            txt = ''.join([f"Turn {state_for_llm["turn"] + i}: {act}\n" for i, act in enumerate(scratch_pad)])
        except Exception as e:
            print(f"scratch_pad: {scratch_pad}")
            print(f"Error formatting scratch pad: {e}")
            exit(0)
        description += f"**Plan Advice**:\n'''\n{txt}\n'''\n"
    description += f"""**Map**:\n'''\n{state_for_llm['map']}\n'''"""
    return description
