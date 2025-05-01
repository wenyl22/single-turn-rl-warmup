import sys
import re
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time 
import pandas as pd
from minatar.environment import Environment
from minatar.environments.freeway import Env
from extract_utils import find_best_match, extract_boxed
from envs.client_utils import LocalThreadedLLMClient, ApiThreadedLLMClient
from vllm import SamplingParams
VLLM_client = None 
seed_mapping = {
    0: (1004, 17, 0), 
    1: (1026, 18, 0), 
    2: (1536, 14, 0), 
    3: (1798, 17, 5), 
    4: (1858, 16, 0), 
    5: (2408, 19, 0), 
    6: (2499, 13, 0), 
    7: (2867, 22, 0)
}
def setup_thread_VLLM_client(token_per_tick, args):
    global VLLM_client
    if args.api_keys == []:
        VLLM_client = LocalThreadedLLMClient(token_per_tick)
    else:
        VLLM_client = ApiThreadedLLMClient(token_per_tick, args) 
   
def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client
    
def freeway_game_loop(log_file, seed, difficulty = 8, max_tokens = 1000):
    """
    Freeway Game Loop in Single-Agent System.
    """
    from prompts.freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION
    client = VLLM_client
    thread_id = client.add_new_thread()
    env = Environment('freeway', sticky_action_prob=0)
    env.env.difficulty = difficulty
    if seed in seed_mapping:
        seed = seed_mapping[seed]
        env.seed(seed[0])
        env.reset()
        for i in range(seed[2]):
            env.act(0)
    else:
        env.seed(seed)
        env.reset()
    start_time = time.time()
    terminal = False
    reward = 0
    game_turn = 0
    logs = {'description': [], 'render':[], 'plan_agent_response': [], 'selected_action': []}
    while True:
        action = 0
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm)
        available_actions_list = [f'{chr(65+i)}. {action}' for i, action in enumerate(state_for_llm['available_actions'])]
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": LLM_BASE_PROMPT + state_description}
        ]
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=max_tokens
        )
        response = client.run_inference(thread_id, messages, STAY_COMPLETION, sampling_params)
        selected_action = find_best_match(client, thread_id, response, available_actions_list, STAY_COMPLETION)
        if "stay" in selected_action.lower():
            action = 0
        elif "up" in selected_action.lower(): 
            action = 2
        elif "down" in selected_action.lower():
            action = 4
        logs['description'].append(state_description)
        logs['render'].append('\n' + env.env.state_string())
        logs["plan_agent_response"].append(response)
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
        print(f"Thread {thread_id} - Game Turn: {game_turn}, Position: {env.env.pos}")
    return {
        'seed': seed,
        'difficulty': difficulty,
        'game_turn': game_turn,
        'game_time': time.time() - start_time
    }

def legacy_ma_freeway_game_loop(log_file, seed, difficulty = 8, max_tokens = 1000):
    """
    Freeway Game Loop in Single-Agent System.
    """
    from prompts.ma_freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION, SUPERVISOR_PORMPT, PLAN_PROMPT, REACT_PROMPT, FOLLOW_PLAN_PROMPT
    client = VLLM_client
    thread_id = client.add_new_thread()
    env = Environment('freeway', sticky_action_prob=0)
    env.env.difficulty = difficulty
    if seed in seed_mapping:
        seed = seed_mapping[seed]
        env.seed(seed[0])
        env.reset()
        for i in range(seed[2]):
            env.act(0)
    else:
        env.seed(seed)
        env.reset()
    reward = 0
    game_turn = 0
    scratch_pad = ""
    start_time = time.time()
    terminal = False
    logs = {'description': [], 'render':[], 'supervisor_response': [],  'selected_agent': [], 'plan_agent_response':[], 'selected_agent_response': [], 'selected_action': []}
    while True:
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm, scratch_pad)
        logs['description'].append(state_description)
        logs['render'].append('\n' + env.env.state_string())
        available_actions_list = [f'{chr(65+i)}. {action}' for i, action in enumerate(state_for_llm['available_actions'])]
        ## call supervisor agent
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": LLM_BASE_PROMPT + SUPERVISOR_PORMPT + state_description}
        ]
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=max_tokens
        )
        response = client.generate(thread_id, messages, sampling_params)['text']
        logs['supervisor_response'].append(response)
        # parse the response
        selected_agent = find_best_match(client, thread_id, response, ["A. Plan Agent", "B. Follow Plan Agent", "C. React Agent"], "C. React Agent")
        logs['selected_agent'].append(selected_agent)
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=max_tokens - 30
        )
        if "plan" in selected_agent.lower() and "follow" not in selected_agent.lower():
            ## call plan agent
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + PLAN_PROMPT + state_description}
            ]
            
            response = client.generate(thread_id, messages, sampling_params)['text']
            logs['plan_agent_response'].append(response)
            scratch_pad = extract_boxed(response)
            state_description = state_to_description(state_for_llm, scratch_pad)
        else:
            # dummy function call, indicating the thread is alive
            _ = client.generate(thread_id, [], sampling_params)
            logs['plan_agent_response'].append("")
        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=max_tokens
        )
        if "plan" in selected_agent.lower():
            ## call follow plan agent
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + FOLLOW_PLAN_PROMPT + state_description}
            ]
            response = client.generate(thread_id, messages, sampling_params)['text']
        else:
            ## call react agent
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + REACT_PROMPT + state_description}
            ]
            response = client.generate(thread_id, messages, sampling_params)['text']
            
        selected_action = find_best_match(client, thread_id, response, available_actions_list, STAY_COMPLETION)
        if "stay" in selected_action.lower():
            action = 0
        elif "up" in selected_action.lower(): 
            action = 2
        elif "down" in selected_action.lower():
            action = 4
        logs["selected_agent_response"].append(response)
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
        print(f"Thread {thread_id} - Game Turn: {game_turn}, Position: {env.env.pos}")
    return {
        'seed': seed,
        'difficulty': difficulty,
        'game_turn': game_turn,
        'game_time': time.time() - start_time
    }

def ma_freeway_game_loop(log_file, seed, difficulty = 8, max_tokens = 1000):
    """
    New version of Freeway Game Loop in Single-Agent System.
    """

    from prompts.ma_freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, PLAN_PROMPT

    # Environment setup
    client = VLLM_client
    thread_id = client.add_new_thread()
    env = Environment('freeway', sticky_action_prob=0)
    env.env.difficulty = difficulty
    if seed in seed_mapping:
        seed = seed_mapping[seed]
        env.seed(seed[0])
        env.reset()
        for i in range(seed[2]):
            env.act(0)
    else:
        env.seed(seed)
        env.reset()

    # Game loop variables
    reward = 0
    game_turn = 0
    scratch_pad = ""
    start_time = time.time()
    terminal = False

    # Log storage
    logs = {'description': [], 'render':[], 'supervisor_response': [], 'plan_agent_response':[], 'scratch_pad': [], 'selected_agent': [], 'selected_action': []}

    while True:
        # Build the state for LLM
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm, scratch_pad)
        logs['description'].append(state_description)
        logs['render'].append('\n' + env.env.state_string())
        log_supervisor_response, log_selected_agent, log_plan_agent_response, log_selected_action = "", "", "", ""

        # 1. Call Plan Agent if the scratch pad is empty
        if scratch_pad == "":
            # Call Plan Agent
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + PLAN_PROMPT + state_description}
            ]
            sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                max_tokens=max_tokens - 30
            )
            log_plan_agent_response = client.run_inference(thread_id, messages, "\\boxed{S}", sampling_params)
            # extract answer in \boxed{}

            match = re.search(r'oxed{([^}]*)}', log_plan_agent_response.split("</think>")[-1])
            if match:
                scratch_pad = match.group(1).strip()
            else:
                scratch_pad = "S"
            # keep only 'U', 'D', 'S' in scratch_pad
            scratch_pad = re.sub(r'[^UDS]', '', scratch_pad.upper())
        if scratch_pad == "":
            scratch_pad = "S"
        logs['scratch_pad'].append(scratch_pad)
        # 2. Check if next action in the scratch pad is valid
        assert scratch_pad[0] in ['U', 'D', 'S']
        collision, selected_action = reaction_to_collision(state_for_llm, scratch_pad[0])
        if collision:
            # Execute the action directly
            action = 0 if 'stay' in selected_action.lower() else 2 if 'up' in selected_action.lower() else 4
            scratch_pad = ""
            log_supervisor_response = f"Collision risk detected, {selected_action}."
            log_selected_agent = "C. React Agent"
            log_selected_action = selected_action
        else:
            # Follow Plan Agent
            action = 0 if scratch_pad[0] == 'S' else 2 if scratch_pad[0] == 'U' else 4
            scratch_pad = scratch_pad[1:]
            log_supervisor_response = "No collision risk detected."
            log_selected_agent = "B. Follow Plan Agent"
            log_selected_action = selected_action
        
        # Save logs to CSV
        logs['supervisor_response'].append(log_supervisor_response)
        logs['selected_agent'].append(log_selected_agent)
        logs['plan_agent_response'].append(log_plan_agent_response)
        logs['selected_action'].append(log_selected_action)
        df = pd.DataFrame(logs)
        df.to_csv(log_file)

        # Execute the action in the environment
        reward, terminal = env.act(action)
        game_turn += 1
        if reward > 0.5:
            print(f"Get to the otherside in {game_turn} actions!")
            break
        if terminal or (game_turn > 100):
            print("Fail to get to the otherside in required turns")
            break
        print(f"Thread {thread_id} - Game Turn: {game_turn}, Position: {9 - env.env.pos}")

    # Return game results
    return {
        'seed': seed,
        'difficulty': difficulty,
        'game_turn': game_turn,
        'game_time': time.time() - start_time
    }

def llm_state_builder(env: Env):
    player_states = 9 - env.pos
    car_states = []
    for car in env.cars:
        # car: [x, y, timer, speed, length]
        if car[3] is None:
            car_states.append((9 - car[1], None, None, None, None))
            continue
        dir = 'left' if car[3] < 0 else 'right'
        speed = int(12 / abs(car[3]))
        pos = 12 * (car[0] - 4)
        if abs(car[3]) >= 1:
            if dir == 'left':
                pos -= (abs(car[3]) - car[2] - 1) * speed
            else:
                pos += (abs(car[3]) - car[2] - 1) * speed
        else:
            pass
        assert car[2] < abs(car[3])
        car_states.append( (9 - car[1], pos, dir, speed, car[4] * 12 - 1) )
    car_states.sort(key=lambda x: x[0])
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

def state_to_description(state_for_llm, scratch_pad = None):
    # (9 - car[1], pos, dir, speed, car[4] * 12 - 1)
    description = f"- Player Position: (0, {state_for_llm['player_states']}).\n"
    if scratch_pad is not None:
        description += f'- Plan Scratch Pad: {scratch_pad if scratch_pad != "" else "Empty"}.\n'
    description += '- Cars on Each Freeway:\n'
    las_car = -1
    num = [0] * 9
    for car in state_for_llm['car_states']:
        if car[1] is not None:
            num[car[0]] += 1
    for car in state_for_llm['car_states']:
        span = car[4] if car[2] == 'left' else -car[4]
        if las_car != car[0]:
            description += f"\t- Freeway {car[0]}: {num[car[0]]} cars.\n"
            las_car = car[0]
        description += f"\t\t - Head at **x = {car[1]}**, tail at **x = {car[1] + span}**, direction = {car[2]}, speed = {car[3]}.\n"
    description += f'- Available actions:\n{get_available_actions(state_for_llm)}'
    return description

def get_available_actions(state_for_llm):
    description = ''
    for i, action in enumerate(state_for_llm['available_actions']):
        description += f'{chr(65+i)}. {action}\n'
    return description

# TODO: Check if the speed should be considered
def reaction_to_collision(state_for_llm, next_action_char):
    """
    Check if there is a collision risk by staying in the same freeway.
    If true, decide to move up or down.
    Else, stay in the same freeway.

    Returns:
        - stay_collision: bool, True if there is a collision risk by staying in the same freeway
        - preferred_action: str, next action to take if there is a collision risk
    """
    # Check if the player is on the same freeway as any car for next turn

    collision = [False, False, False] # [stay, up, down]
    next_action_ind = 0 if next_action_char == 'S' else 1 if next_action_char == 'U' else 2
    # corner case: if the player is on freeway 0
    if state_for_llm['player_states'] == 0:
        collision[2] = True

    for car in state_for_llm['car_states']:
        # (9 - car[1], pos, dir, speed, car[4] * 12 - 1)
        head = car[1]
        span = car[4] if car[2] == 'left' else -car[4]
        tail =  head + span
        
        # Change the range of x-values to check for collision
        if car[2] == 'left':
            head = head - car[3]
            tail = tail - car[3]
        else:
            head = head + car[3]
            tail = tail + car[3]
        
        # Check if the player is on the same freeway as any car for next turn
        if car[0] == state_for_llm['player_states']:
            if head <= 0 <= tail or tail <= 0 <= head:
                collision[0] = True
                
        # Check if the player is on the freeway above the car
        if car[0] == state_for_llm['player_states'] + 1:
            if head <= 0 <= tail or tail <= 0 <= head:
                collision[1] = True
        
        # Check if the player is on the freeway below the car
        if car[0] == state_for_llm['player_states'] - 1:
            if head <= 0 <= tail or tail <= 0 <= head:
                collision[2] = True
    
    perfer_action_ind = next_action_ind
    if collision[next_action_ind]:
        for i in range(3):
            if not collision[i]: 
                perfer_action_ind = i
                break
    available_action_list = ["Stay in the same freeway", "Move up to Freeway " + str( state_for_llm['player_states'] + 1), "Move down to Freeway " + str(state_for_llm['player_states'] - 1)]
    return collision[next_action_ind], available_action_list[perfer_action_ind]
