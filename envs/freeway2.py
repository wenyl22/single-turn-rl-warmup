import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import time 
import pandas as pd
from minatar.environment import Environment
from minatar.environments.freeway import Env
from envs.utils.extract_utils import extract_scratch_pad, extract_boxed
from envs.utils.client_utils import ApiThreadedLLMClient
from vllm import SamplingParams
from envs.prompts.sa_freeway_math import CAR_STATE, LLM_SYSTEM_PROMPT, STAY_COMPLETION
VLLM_client = None 
seed_mapping = {
    0: (1000, 13, 0), 1: (1001, 11, 0), 2: (1002, 11, 0), 3: (1003, 11, 0),
    4: (1013, 13, 0), 5: (1014, 12, 0), 6: (1016, 11, 0), 7: (1018, 11, 0)
}
def setup_thread_VLLM_client(token_per_tick, args):
    global VLLM_client
    VLLM_client = ApiThreadedLLMClient(token_per_tick, args) 
   
def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client

def game_loop(log_file, seed, args, thread_id):
    client = VLLM_client
    from envs.prompts.ma_freeway_math import MATH_PROMPT as LLM_BASE_PROMPT, MATH_PROMPT_LOW_LEVEL
    from envs.prompts.ma_freeway_game import ORIGINAL_ANSWER_FORMAT as LLM_ANSWER_FORMAT
    client.add_new_thread(thread_id)
    env = Environment('freeway', sticky_action_prob=0)
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
                {"role": "user", "content": LLM_BASE_PROMPT + LLM_ANSWER_FORMAT + state_description}
            ]
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new_tokens - 5)
        # OPTION2: Interrupt the thread with new state.
        if args.budget_forcing == "si":
            sampling_params.max_tokens = client.token_per_tick - 5
            end, log_plan_agent_response = client.run_inference_with_interruption(thread_id, messages, "", sampling_params)
            if end:
                scratch_pad = extract_scratch_pad(log_plan_agent_response, scratch_pad, valid_actions='SUD')
        # OPTION1: Automatically drop message if the thread is planning state for previous turns.
        else:
            turns = client.token_queue_len[thread_id] // client.token_per_tick
            # The message will be automatically dropped if the thread is planning state for previous turns.
            log_plan_agent_response = client.run_inference(thread_id, messages, "", sampling_params)
            if log_plan_agent_response != "": # agent responds with a plan
                scratch_pad = extract_scratch_pad(log_plan_agent_response, scratch_pad, valid_actions='SUD')
                scratch_pad = scratch_pad[turns:] if turns < len(scratch_pad) else ""
        logs['plan_agent_response'].append(log_plan_agent_response)
        if scratch_pad == "":
            scratch_pad = "U"
        logs['scratch_pad'].append(scratch_pad)
        ### --- Low Level Agent --- ###
        available_action_list = ["Stay in the same freeway", "Move up to Freeway " + str( state_for_llm['player_states'] + 1), "Move down to Freeway " + str(state_for_llm['player_states'] - 1)]
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
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
            log_supervisor_response = client.run_low_level_inference(thread_id, messages, sampling_params)
            action = extract_boxed(log_supervisor_response)
        log_selected_action = available_action_list[0 if action == 'S' else 1 if action == 'U' else 2]
        if action == scratch_pad[0]:
            log_selected_agent = "B. Follow Plan Agent"
            scratch_pad = scratch_pad[1:]
        else:
            log_selected_agent = "C. React Agent"
            scratch_pad = ""
        action = 0 if action == 'S' else 2 if action == 'U' else 4
        #     safe = not supervise_collision(state_for_llm, scratch_pad[0])
        # if safe: # Follow Plan
        #     log_selected_agent = "B. Follow Plan Agent"
        #     log_supervisor_response = "Safe to follow plan."
        #     action = 0 if scratch_pad[0] == 'S' else 2 if scratch_pad[0] == 'U' else 4
        #     scratch_pad = scratch_pad[1:]
        #     selected_action = 0 if action == 0 else 1 if action == 2 else 2
        #     log_selected_action = available_action_list[selected_action]
        # else:
        #     log_selected_agent = "C. React Agent"
        #     log_supervisor_response = "Plan leads to immediate collision, react."
        #     selected_action = available_action_list[react_to_collision(state_for_llm)]
        #     action = 0 if 'Stay' in selected_action else 2 if 'up' in selected_action else 4
        #     scratch_pad = ""
        #     log_selected_action = selected_action
        logs['supervisor_response'].append(log_supervisor_response)
        logs['selected_agent'].append(log_selected_agent)
        logs['selected_action'].append(log_selected_action)
        df = pd.DataFrame(logs)
        df.to_csv(log_file)
        r, terminal = env.act(action)
        game_turn += 1
        reward += r
        if r > 0.5:
            print(f"Thread {thread_id} Get to the otherside in {game_turn} actions!")
            break
        elif r < 0:
            # reset, drop the message in the queue
            print(f"Thread {thread_id} Hit by a car, reset the game.")
            env.env.pos = 9
            env.seed(seed[0])
            env.reset()
            for i in range(seed[2]):
                env.act(0)
            while client.token_queue_len[thread_id] > 0:
                _ = client.run_inference(thread_id, [], "", None)
            scratch_pad = ""
        if terminal or (game_turn > 100):
            print("Fail to get to the otherside in required turns")
            break
        print(f"Thread {thread_id} - Game Turn: {game_turn}, Position: {9 - env.env.pos}")

    return {
        'seed': seed,
        'game_turn': game_turn,
        'reward': reward,
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

def state_to_description(state_for_llm, state_prediction = 0, scratch_pad = None):
    description = ""
    if state_prediction == 0:
        description += f"### **Game State**\n**Current Turn Player Position:** \((0, {state_for_llm['player_states']})\) \n"
        if scratch_pad is not None:
            description += f"**Plan Advice**: {",".join(scratch_pad)}\n"
        description += f"**Current Turn Car State**:\n"
    else:
        description += f"**Predicted Car State After {state_prediction} Turns**:\n"
    description += CAR_STATE
    car_info = ""
    lane = 1
    for car in state_for_llm['car_states']:
        if car[0] != lane:
            description += f"| {lane} | \({car_info}\) |\n"
            car_info = ""
            lane = car[0]
        span = car[4] if car[2] == 'left' else -car[4]
        if car_info != "":
            car_info += ", "
        car_info += f"({car[1]}, {car[1] + span}, {car[2]}, {car[3]})"
    description += f"| {lane} | \({car_info}\) |\n"
    return description

def get_available_actions(state_for_llm):
    description = ''
    for i, action in enumerate(state_for_llm['available_actions']):
        description += f'{chr(65+i)}. {action}\n'
    return description


def supervise_collision(state_for_llm, scratch_pad, future_step = 3):
    """
    Check if there is a collision risk by following "scratch_pad" in the next "future_step" turns.
    future_step = min(future_step, len(scratch_pad))
    """
    future_step = min(future_step, len(scratch_pad))
    pos = state_for_llm['player_states']
    for t in range(future_step):
        pos += 1 if scratch_pad[t] == 'U' else -1 if scratch_pad[t] == 'D' else 0
        if pos == 9:
            return False
        pos = max(0, pos)
        for car in state_for_llm['car_states']:
            if car[0] != pos:
                continue
            head = car[1]
            span = car[4] if car[2] == 'left' else -car[4]
            tail =  head + span        
            if car[2] == 'left':
                head = head - car[3] * (t + 1)
                tail = tail - car[3] * (t + 1)
            else:
                head = head + car[3] * (t + 1)
                tail = tail + car[3] * (t + 1)
            if head <= 0 <= tail or tail <= 0 <= head:
                return True
    return False

def react_to_collision(state_for_llm):
    """
    Returns:
        - stay_collision: bool, True if there is a collision risk by staying
        - preferred_action: str, next action to take if there is a collision risk
    """
    # Check if the player is on the same freeway as any car for next turn
    collision = [
        supervise_collision(state_for_llm, 'S'),
        supervise_collision(state_for_llm, 'U'),
        supervise_collision(state_for_llm, 'D')
    ] # [stay, up, down]
    # corner case: if the player is on freeway 0
    if state_for_llm['player_states'] == 0:
        collision[2] = True    
    perfer_action_ind = 0
    for i in [1, 0, 2]:
        if not collision[i]: 
            perfer_action_ind = i
            break
    return perfer_action_ind

def check_collision(state, X, action):
    # whether there must be a collision in X steps, no matter what action is taken
    pos = state['player_states']
    pos += 1 if action == 'U' else -1 if action == 'D' else 0
    if pos == 9:
        return False
    def car_collision(car, t, p):
        if car[0] != p:
            return False
        head = car[1]
        span = car[4] if car[2] == 'left' else -car[4]
        tail =  head + span        
        if car[2] == 'left':
            head = head - car[3] * (t + 2)
            tail = tail - car[3] * (t + 2)
        else:
            head = head + car[3] * (t + 2)
            tail = tail + car[3] * (t + 2)
        return head <= 0 <= tail or tail <= 0 <= head
    live_pos = set()
    live_pos.add(pos)
    for t in range(X):
        # print(f"a: {action}, t: {t}, live_pos: {live_pos}")
        new_live_pos = set()
        for pos in live_pos:
            temp = pos
            for a in ['S', 'U', 'D']:
                temp = pos + (1 if a == 'U' else -1 if a == 'D' else 0)
                if temp == 9:
                    return False
                flag = False
                for car in state['car_states']:
                    if car_collision(car, t, temp):
                        flag = True
                        break
                if not flag:
                    new_live_pos.add(temp)
        if len(new_live_pos) == 0:
            return True
        live_pos = new_live_pos
    return False   

def react_to_collision(state_for_llm, X = 0):
    """
    Returns:
        - stay_collision: bool, True if there is a collision risk by staying
        - preferred_action: str, next action to take if there is a collision risk
    """
    # Check if the player is on the same freeway as any car for next turn
    collision = [
        supervise_collision(state_for_llm, 'S'),
        supervise_collision(state_for_llm, 'U'),
        supervise_collision(state_for_llm, 'D')
    ] # [stay, up, down]
    # corner case: if the player is on freeway 0
    if state_for_llm['player_states'] == 0:
        collision[2] = True    
    perfer_action_ind = 0
    for i in [1, 0, 2]:
        if not collision[i] and not check_collision(state_for_llm, X, 'SUD'[i]):
            perfer_action_ind = i
            break        
    return perfer_action_ind

