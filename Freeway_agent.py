import time 
import os 
from openai import OpenAI, AzureOpenAI
import datetime 
import re 
from fuzzywuzzy import process 
import numpy as np 
import pandas as pd
from minatar.environments.freeway import Env
import threading
import queue
class LocalThreadedLLMClient:
    def __init__(self):
        self.num_threads = 0
        self.lock = threading.Lock()
        self.query_queues = {}
        self.response_queues = {}

    def add_new_thread(self):
        self.lock.acquire()
        self.num_threads += 1
        thread_id = self.num_threads
        self.query_queues[thread_id] = queue.Queue()
        self.response_queues[thread_id] = queue.Queue()
        self.lock.release()
        return thread_id
    
    def generate(self, thread_id, prompt):
        self.query_queues[thread_id].put(prompt)
        return self.response_queues[thread_id].get()

VLLM_client = None # this will be lazy loaded

def setup_thread_VLLM_client():
    global VLLM_client
    VLLM_client = LocalThreadedLLMClient()

def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client

def add_to_dict_list(dictionary, key, item):
    if key not in dictionary:
        dictionary[key] = [item]
    else:
        dictionary[key].append(item)


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
global_llm_model = None
global_tokenizer = None

STAY_COMPLETION = f"""Action: Stay in the same freeway"""

class LLMManager:
    def __init__(self, model_name, cache_dir, temperature=0.6, do_sample=True, max_new_tokens=2000, top_p=0.9, token_per_tick = 500, run_type="serial"):
        global global_llm_model
        global global_tokenizer
        global VLLM_client
        self.model_name = model_name 
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        
        self.token_per_tick = token_per_tick
        self.token_queue_len = 0
        self.accum = 0
        self.run_type = run_type
        if self.run_type == "thread":
            assert VLLM_client is not None
            self.client = VLLM_client
            self.thread_id = self.client.add_new_thread()
        else:
            if global_llm_model is None:
                global_llm_model = LLM(model_name)
        global_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def run_openai_inference(self, messages):
        self.accum += self.token_per_tick
        if self.token_queue_len > 0:
            if self.token_queue_len <= self.accum:
                self.accum = 0
                self.token_queue_len = 0
                return self.resp
            else:
                return STAY_COMPLETION + "!!"
        api_call_start = time.time()
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens
        )
        input_text = global_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if "deepseek" not in self.model_name:
            input_text += "<think>"
        print(f"{bcolors.OKCYAN}INPUT TEXT: {input_text}{bcolors.ENDC}")
        
        if self.run_type == "serial":        
            response = global_llm_model.generate(input_text, sampling_params)
            self.resp = response[0].outputs[0].text
            self.token_queue_len = len(response[0].outputs[0].token_ids)

        else:
            response = self.client.generate(self.thread_id, input_text)
            self.resp = response['text']
            self.token_queue_len = len(response['token_ids'])

        print(f"{bcolors.FAIL}LLM INFERENCE TIME: {time.time() - api_call_start}{bcolors.ENDC}")
        if self.accum >= self.token_queue_len:
            self.accum = 0
            self.token_queue_len = 0
            return self.resp
        else:
            return STAY_COMPLETION + "!!"
def prompt_builder(env: Env):
    player_states = (9 - env.pos, "You can move in this turn" if env.move_timer == 0 else f"You cannot move in the following {env.move_timer} turns")
    car_states = []
    for car in env.cars:
        # dir = "towards"
        # if car[0] > 4 and car[3] > 0:
        #     dir = "away from"
        # elif car[0] < 4 and car[3] < 0:
        #     dir = "away from"
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
    # reverse the order of cars
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


class LLMAgent:
    def __init__(self, model, max_new_tokens=2000, token_per_tick=500, run_type="serial", seed=None):
        self.max_new_tokens = max_new_tokens
        self.token_per_tick = token_per_tick
        self.DEBUG = False     
        self.enable_cache = False # True    
        self.write_to_cache = False 
        self.save_trajectory = True # True 
        self.log_csv_dict = {}
        self.model = model 
        self.run_type = run_type

        self.llm = LLMManager(model_name=self.model, cache_dir=os.getenv('HF_HOME'), max_new_tokens=self.max_new_tokens, token_per_tick=self.token_per_tick, run_type=self.run_type)
        
        ### LOGGING ###
        self.time_stamp = seed if seed is not None else datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print("max_new_tokens: ", self.max_new_tokens)
        print("token_per_tick: ", self.token_per_tick)
        print("Time stamp: ", self.time_stamp)
        
        self.log_dir = f'logs/freeway/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)            
        self.log_dir = f'logs/freeway/{model.split('/')[-1]}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        ### LOGGING ###
        
        self.player_actions = []
        self.llm_system_prompt = '''Please reason step by step, and put your final answer within \\boxed{}.'''
        self.llm_base_prompt = '''
# **Freeway Game: Optimal Action Selection**  

## **Game Overview**  
You are playing **"Freeway,"** a game where you must guide your character safely across multiple lanes of moving traffic. Your goal is to **reach the destination (y = 9) from the starting point (y = 0) in the fewest turns while avoiding collisions with cars.**  

Imagine a **2D grid** where:  
- **The vertical axis (y)** represents different freeways, numbered from `0` to `9`.  
- **The horizontal axis (x)** represents positions along each freeway.  

You always stay at **x = 0**, meaning you cannot move left or right. Instead, your only movement options are:  
- Moving **up** (to a higher freeway, y → y + 1).  
- Moving **down** (to a lower freeway, y → y - 1).  
- Staying **on the same freeway** (no movement).  

## **Game Mechanics**  

### **1. Freeways & Cars**  
Each freeway (from y = 1 to y = 8) has cars moving left or right.  
- **Cars move at a fixed speed per turn** (e.g., a speed of 3 means the car moves 3 units in the x-direction each turn).  
- **Each car has a span**, which is the range of x-values it occupies.  
- **Movement Direction**:  
  - If a car moves **right**, its span extends as it moves forward.  
  - If a car moves **left**, its span moves backward.  

**Example:**  
A car on Freeway 2 with a **head position at x = 18** and a **tail position at x = 29** moves **left at a speed of 6**.  
- After **one turn**, its new span will be **[head: 12, tail: 23]**.  
- After **two turns**, its span will be **[head: 6, tail: 17]**.  

### **2. Collisions**  
A **collision happens if, at any point before or after your move, your position (x = 0, y) overlaps with a car’s span**.  
- If a collision occurs, you are **reset to the starting position (0, 0)**.  
- To avoid collisions, you must predict car movements and time your actions carefully.  

## **Game State Representation**  
Each turn, you receive the **current game state**, which includes:  
- **Your current position**: (x = 0, y).  
- **Car information for each freeway (y = 1 to 8)**:  
  - **Head position** (front of the car).  
  - **Tail position** (back of the car).  
  - **Direction** (left or right).  
  - **Speed** (how many x-units the car moves per turn).  

## **Your Task: Find the Best Move**  
Each turn, you must **analyze car positions, predict future movements, and decide the safest and most efficient action** from the following options:  
- **A**: Move **up** to Freeway (y + 1).  
- **B**: Move **down** to Freeway (y - 1).  
- **C**: Stay on the current freeway. '''
            
        self.tokens_used = 0 

        # print(self.all_actions)
        self.log_csv_dict = {}
        self.action_history = []
    def find_best_match(self, action_string):
        if "</think>" not in action_string:
            action_string = STAY_COMPLETION
        else:
            action_string = action_string.split("</think>")[-1]
        if action_string == "":
            action_string = STAY_COMPLETION
        # search for \boxed{} and extract the content
        match = re.search(r'\\boxed\{(.+?)\}', action_string)
        if match:
            selected_match = match.group(1).strip()
        else:
            selected_match = action_string
        for action in self.available_actions_list:
            if selected_match.lower() in action.lower():
                return action 
        selected_move, score = process.extractOne(selected_match, self.available_actions_list)
        return selected_move

    def _state_to_description(self, state_for_llm):
        description = f"-**Your position**: (0, {state_for_llm["player_states"][0]}).\n"
        description += '-**Cars on each freeway**:\n'
        for car in state_for_llm['car_states']:
            span = 11 if car[2] == 'left' else -11
            description += f"\t-**Freeway {car[0]}**: head at **x = {car[1]}**, tail at **x = {car[1] + span}**, direction = {car[2]}, speed = {car[3]}.\n"
        description += f'Available actions:\n{self._get_available_actions(state_for_llm)}'
        return description
    def _get_available_actions(self, state_for_llm):
        self.available_actions_list = []
        description = ''
        for i, action in enumerate(state_for_llm['available_actions']):
            self.available_actions_list.append(f'{chr(65+i)} {action}')
        for action in self.available_actions_list:
            description += f'{action}\n'
        return description
    def get_next_move(self, state_for_llm):
        
        state_description = self._state_to_description(state_for_llm)
        print(f"{bcolors.OKBLUE}{state_description}{bcolors.ENDC}")
        messages = [
            {"role": "system", "content": self.llm_system_prompt},
            {"role": "user", "content": self.llm_base_prompt + state_description}
        ]
        add_to_dict_list(self.log_csv_dict, 'state_description', state_description)

        action_string = self.llm.run_openai_inference(messages=messages)
        print(f"{bcolors.OKGREEN}LLM Response: {action_string}{bcolors.ENDC}")
        add_to_dict_list(self.log_csv_dict, 'llm_response', action_string)
        if "</think>" in action_string:
            if 'boxed' in action_string:
                add_to_dict_list(self.log_csv_dict, 'action_string', action_string.split("</think>")[-1].split("\\boxed{")[-1].split("}")[0])
            else:
                add_to_dict_list(self.log_csv_dict, 'action_string', action_string.split("</think>")[-1])
        else:
            add_to_dict_list(self.log_csv_dict, 'action_string', STAY_COMPLETION)
        selected_action = self.find_best_match(action_string)
        if selected_action in self.available_actions_list:
            selected_move_idx = self.available_actions_list.index(selected_action)
        print("SELECTED ACTION: ", state_for_llm['available_actions'][selected_move_idx])
        add_to_dict_list(self.log_csv_dict, 'selected_action', selected_action)
        df = pd.DataFrame(self.log_csv_dict)
        df.to_csv(f"{self.log_dir}/{self.max_new_tokens}_{self.token_per_tick}_{self.time_stamp}.csv") 
        return state_for_llm['available_actions'][selected_move_idx]