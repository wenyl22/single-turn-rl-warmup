import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time 
import pandas as pd
from minatar.environment import Environment
from minatar.environments.freeway import Env
from prompts.freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION
from prompts.eval import EVAL_PROMPT, FEW_SHOT_EXAMPLES
from utils import LocalThreadedLLMClient
VLLM_client = None 

def setup_thread_VLLM_client(token_per_tick):
    global VLLM_client
    VLLM_client = LocalThreadedLLMClient(token_per_tick=token_per_tick)

def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client

def freeway_game_loop(log_file, seed, difficulty = 8):
    """
    Freeway Game Loop in Single-Agent System.
    """
    from prompts.freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION
    client = VLLM_client
    thread_id = client.add_new_thread()
    env = Environment('freeway', sticky_action_prob=0)
    env.env.difficulty = difficulty
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
            
            # utils
            import re
            
            def find_first_capital(s):
                match = re.search(r'[A-Z]', s)
                if match:
                    return match.group()
                return ""
            def find_boxed(s):
                match = re.search(r'\\boxed\{(.+?)\}', s)
                if match:
                    return match.group(1).strip()
                return ""
                    
            # extract a boxed capital letter (like "\boxed{A}" or "\boxed{B}") if present
            if "</think>" in response:
                response = response.split("</think>")[-1]
                
            # Extract capital letter `selected_match` for next move
            
            
            # Raw Extraction
            selected_match = find_boxed(response) 
            for action_option in available_actions_list:
                if selected_match in action_option:
                    selected_match = action_option[0]
                    break
            selected_match = find_first_capital(selected_match) or selected_match
            # print("Raw extraction result: ", selected_match)
            if not (selected_match.isalpha() and ord(selected_match) - ord('A') < len(available_actions_list)):
                # Model Extraction
                extract_answer_messages = [
                    {"role": "system", "content": EVAL_PROMPT}
                ]
                # extract_answer_messages += FEW_SHOT_EXAMPLES
                extract_answer_messages.append(
                    {"role": "user", "content": f"action_string: \"{response}\"\navailable_actions_list: {available_actions_list}"}
                )
                selected_response = client.run_inference(thread_id, extract_answer_messages, STAY_COMPLETION, True)
                # print(f"extract_answer_message: {extract_answer_messages[-1]['content']}")
                # print(f"selected_response: {selected_response}")
                
                selected_match = find_boxed(response) or selected_match
                selected_match = find_first_capital(selected_match) or selected_match
            
            if selected_match.isalpha() and ord(selected_match) - ord('A') < len(available_actions_list):
                selected_action = available_actions_list[ord(selected_match) - ord('A')]
            else:
                selected_action = available_actions_list[-1]
            
            # TODO: Apply to other envs.
            
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
        'difficulty': difficulty,
        'game_turn': game_turn,
        'game_time': time.time() - start_time
    }

def ma_freeway_game_loop(log_file, seed, difficulty = 8):
    """
    Freeway Game Loop in Single-Agent System.
    """
    from prompts.ma_freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION, SUPERVISOR_PORMPT, PLAN_PROMPT, REACT_PROMPT, FOLLOW_PLAN_PROMPT
    client = VLLM_client
    thread_id = client.add_new_thread()
    env = Environment('freeway', sticky_action_prob=0)
    env.env.difficulty = difficulty
    env.seed(seed)
    env.reset()
    reward = 0
    game_turn = 0
    scratch_pad = ""
    start_time = time.time()
    terminal = False
    logs = {'description': [], 'render':[], 'supervisor_response': [], 'selected_agent': [],  'selected_agent_response': [], 'selected_action': []}
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
        print("Supervisor message:", messages[-1]["content"])
        response = client.generate(thread_id, messages)['text']
        print("Supervisor response:", response)
        logs['supervisor_response'].append(response)
        # parse the response
        selected_agent = find_best_match(client, thread_id, response, ["A. Plan Agent", "B. Follow Plan Agent", "C. React Agent"], "C. React Agent")
        logs['selected_agent'].append(selected_agent)
        if "plan" in selected_agent.lower() and "follow" not in selected_agent.lower():
            ## call plan agent
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + PLAN_PROMPT + state_description}
            ]
            response = client.generate(thread_id, messages)['text']
            scratch_pad = extract_boxed(response)
        if "plan" in selected_agent.lower():
            ## call follow plan agent
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + FOLLOW_PLAN_PROMPT + state_description}
            ]
            response = client.generate(thread_id, messages)['text']
        else:
            ## call react agent
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT + REACT_PROMPT + state_description}
            ]
            response = client.generate(thread_id, messages)['text']
            
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
        if car[3] is None:
            car_states.append((9 - car[1], None, None, None))
            continue
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

def state_to_description(state_for_llm, scratch_pad = None):
    description = f"- Player Position: (0, {state_for_llm['player_states']}).\n"
    if scratch_pad is not None:
        description += f'- Plan Scratch Pad: {scratch_pad if scratch_pad != "" else "Empty"}.\n'
    description += '- Cars on each freeway:\n'
    for car in state_for_llm['car_states']:
        if car[3] is None:
            description += f"\t- Freeway {car[0]}: No car on this freeway.\n"
            continue
        span = 11 if car[2] == 'left' else -11
        description += f"\t- Freeway {car[0]}: head at **x = {car[1]}**, tail at **x = {car[1] + span}**, direction = {car[2]}, speed = {car[3]}.\n"
    description += f'- Available actions:\n{get_available_actions(state_for_llm)}'
    return description

def get_available_actions(state_for_llm):
    description = ''
    for i, action in enumerate(state_for_llm['available_actions']):
        description += f'{chr(65+i)}. {action}\n'
    return description

