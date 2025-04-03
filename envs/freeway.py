import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time 
import pandas as pd
from minatar.environment import Environment
from minatar.environments.freeway import Env
from prompts.freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION
from utils import find_best_match, LocalThreadedLLMClient
VLLM_client = None 

def setup_thread_VLLM_client(token_per_tick):
    global VLLM_client
    VLLM_client = LocalThreadedLLMClient(token_per_tick=token_per_tick)

def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client

def freeway_game_loop(log_file, seed):
    client = VLLM_client
    assert client is not None, "VLLM client is not initialized. Please call setup_thread_VLLM_client() first."
    thread_id = client.add_new_thread()
    print("Thread ID:", thread_id, "for seed:", seed, "and log file:", log_file)
    env = Environment('freeway', sticky_action_prob=0)
    env.seed(seed)
    env.reset()
    start_time = time.time()
    terminal = False
    reward = 0
    game_turn = 0
    logs = {'description': [],  'llm_response': [], 'render':[], 'selected_action': []}
    while True:
        action = 0
        if env.env.move_timer == 0:
            state_for_llm = llm_state_builder(env.env)
            state_description = state_to_description(state_for_llm)
            available_actions_list = [f'{chr(65+i)}. {action}' for i, action in enumerate(state_for_llm['available_actions'])]
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + state_description}
            ]
            response = client.run_inference(thread_id, messages, STAY_COMPLETION)
            selected_action = find_best_match(response, available_actions_list, STAY_COMPLETION)
            if "stay" in selected_action.lower():
                action = 0
            elif "up" in selected_action.lower(): 
                action = 2
            elif "down" in selected_action.lower():
                action = 4
            logs['description'].append(state_description)
            logs["llm_response"].append(response)
            logs['render'].append('\n' + env.env.state_string())
            logs["selected_action"].append(selected_action)
        df = pd.DataFrame(logs)
        df.to_csv(log_file)
        reward, terminal = env.act(action)
        game_turn += 1
        if reward > 0.5:
            print(f"Get to the otherside in {game_turn} actions!")
            break
        if terminal or (game_turn > 100):
            print("Fail to get to the otherside in required turns")
            break
    return {
        'seed': seed,
        'game_turn': game_turn,
        'game_time': time.time() - start_time
    }


def llm_state_builder(env: Env):
    player_states = 9 - env.pos
    car_states = []
    for car in env.cars:
        dir = 'left' if car[3] < 0 else 'right'
        speed = 12 // abs(car[3])
        pos = 12 * (car[0] - 4)
        if dir == 'left':
            pos -= (abs(car[3]) - car[2] - 1) * speed
        else:
            pos += (abs(car[3]) - car[2] - 1) * speed
        assert car[2] < abs(car[3])
        car_states.append(
            (9 - car[1], pos, dir, speed)
        )
    car_states = car_states[::-1]
    available_actions = []

    if env.move_timer == 0:
        assert env.pos > 0
        available_actions.append("Move up to Freeway " + str(9 - env.pos + 1))
        if env.pos < 9:
            available_actions.append("Move down to Freeway " + str(9 - env.pos - 1))
    available_actions.append("Stay in the same freeway")
    state_for_llm = {
        'player_states': player_states,
        'car_states': car_states,
        'available_actions': available_actions
    }
    return state_for_llm

def state_to_description(state_for_llm):
    description = f"-**Your position**: (0, {state_for_llm['player_states']}).\n"
    description += '-**Cars on each freeway**:\n'
    for car in state_for_llm['car_states']:
        span = 11 if car[2] == 'left' else -11
        description += f"\t-**Freeway {car[0]}**: head at **x = {car[1]}**, tail at **x = {car[1] + span}**, direction = {car[2]}, speed = {car[3]}.\n"
    description += f'Available actions:\n{get_available_actions(state_for_llm)}'
    return description

def get_available_actions(state_for_llm):
    description = ''
    for i, action in enumerate(state_for_llm['available_actions']):
        description += f'{chr(65+i)}. {action}\n'
    return description

