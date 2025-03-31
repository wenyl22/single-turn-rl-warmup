import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time 
import pandas as pd
from minatar.environment import Environment
from minatar.environments.asterix import Env
from prompts.asterix import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION
from utils import find_best_match, LocalThreadedLLMClient
VLLM_client = None 

def setup_thread_VLLM_client(token_per_tick):
    global VLLM_client
    VLLM_client = LocalThreadedLLMClient(token_per_tick=token_per_tick)

def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client

def asterix_game_loop(log_file, seed):
    client = VLLM_client
    assert client is not None, "VLLM client is not initialized. Please call setup_thread_VLLM_client() first."
    thread_id = client.add_new_thread()
    print("Thread ID:", thread_id, "for seed:", seed, "and log file:", log_file)
    env = Environment('asterix', sticky_action_prob=0)
    env.seed(seed)
    env.reset()
    start_time = time.time()
    terminal = False
    game_turn = 0
    game_reward = 0
    logs = {'description': [], 'llm_response': [], 'selected_action': []}
    while True:
        action = 0
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm)
        available_actions_list = [f'{chr(65+i)} {action}' for i, action in enumerate(state_for_llm['available_actions'])]
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": LLM_BASE_PROMPT + state_description}
        ]
        x, y = state_for_llm['player_states']
        response = client.run_inference(thread_id, messages, STAY_COMPLETION.format(x=x, y=y))
        selected_action = find_best_match(response, available_actions_list, STAY_COMPLETION.format(x=x, y=y))
        if "stay" in selected_action.lower():
            action = 0
        elif "up" in selected_action.lower(): 
            action = 2
        elif "down" in selected_action.lower():
            action = 4
        elif "left" in selected_action.lower():
            action = 1
        elif "right" in selected_action.lower():
            action = 3
            
        logs['description'].append(state_description)
        logs["llm_response"].append(response)
        logs["selected_action"].append(selected_action)
        df = pd.DataFrame(logs)
        df.to_csv(log_file)
            
        reward, terminal = env.act(action)
        game_turn += 1
        game_reward += reward
        if terminal or (game_turn > 1000):
            print(f"Game ends in {game_turn} turns with reward {game_reward}.")
            break
    return {
        'seed': seed,
        'game_reward': game_reward,
        'game_turn': game_turn,
        'game_time': time.time() - start_time
    }


def llm_state_builder(env: Env):
    player_states = (env.player_x, env.player_y)
    entity_states = []
    for entity in env.entities:
        #self.entities[slot] = [x,slot+1,lr,is_gold]
        # (x, slot + 1), lr: True for right, is_gold: True for treasure, False for monster
        if entity is not None:
            entity_states.append((entity[0], entity[1], 'right' if entity[2] else 'left', 'treasure' if entity[3] else 'monster'))
    available_actions = []
    if env.player_x > 0:
        available_actions.append(f"Move left to {env.player_x - 1, env.player_y}")
    if env.player_x < 9:
        available_actions.append(f"Move right to {env.player_x + 1, env.player_y}")
    if env.player_y > 1:
        available_actions.append(f"Move up to {env.player_x, env.player_y - 1}")
    if env.player_y < 8:
        available_actions.append(f"Move down to {env.player_x, env.player_y + 1}")
    available_actions.append(f"Stay at {env.player_x, env.player_y}")
    move_history = env.move_history
    state_for_llm = {
        'current_turn': env.turn,
        'player_states': player_states,
        'entity_states': entity_states,
        'move_history': move_history,
        'available_actions': available_actions
    }
    return state_for_llm

def state_to_description(state_for_llm):
    description = f"**Current turn**: t = {state_for_llm['current_turn']}\n"
    description += f"-**Your position**: (x = {state_for_llm['player_states'][0]}, y = {state_for_llm['player_states'][1]})\n"
    description += '-**Entity information**:\n'
    for i, entity in enumerate(state_for_llm['entity_states']):
        description += f"  - {entity[3]}: Position = ({entity[0]}, {entity[1]}), Direction = {entity[2]}\n"
    if len(state_for_llm['entity_states']) == 0:
        description += "  - No entity exists now.\n"
    description += '-**Entity movement history**: All treasures and monsters moved one unit in their direction on turns ' + ', '.join([str(x) for x in state_for_llm['move_history']]) + '\n'
    description += f'Available actions:\n{get_available_actions(state_for_llm)}'

    return description

def get_available_actions(state_for_llm):
    available_actions_list = []
    description = ''
    for i, action in enumerate(state_for_llm['available_actions']):
        available_actions_list.append(f'{chr(65+i)} {action}')
    for action in available_actions_list:
        description += f'{action}\n'
    return description

