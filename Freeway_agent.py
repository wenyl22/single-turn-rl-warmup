import time 
import os 
from openai import OpenAI, AzureOpenAI
import datetime 
import re 
from fuzzywuzzy import process 
import numpy as np 
import pandas as pd
from minatar.environments.freeway import Env
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
    def __init__(self, model_name, cache_dir, temperature=0.6, do_sample=True, max_new_tokens=2000, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, api_server=True, token_per_tick = 500):
        global global_llm_model
        global global_tokenizer
        self.model_name = model_name 
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.api_server = api_server
        self.device = 'cuda'
        
        self.token_per_tick = token_per_tick
        self.token_queue_len = 0
        self.accum = 0
        if global_llm_model is None:
            global_llm_model = LLM(model_name, device = self.device)
            global_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.inference_fn = self.run_openai_inference                    

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
        # print("input_text:", input_text)
        response = global_llm_model.generate(input_text, sampling_params)

        print(f"{bcolors.FAIL}LLM INFERENCE TIME: {time.time() - api_call_start}{bcolors.ENDC}")
        self.resp = response[0].outputs[0].text
        self.token_queue_len = len(response[0].outputs[0].token_ids)
        if self.accum >= self.token_queue_len:
            self.accum = 0
            self.token_queue_len = 0
            return self.resp
        else:
            return STAY_COMPLETION + "!!"
def prompt_builder(env: Env):
    player_states = (env.pos, "I can move in this turn" if env.move_timer == 0 else f"I cannot move in the following {env.move_timer} turns")
    car_states = []
    for car in env.cars:
        # car[1]: y position
        # car[0]: x position
        # car[2]: speed
        # car[3]: direction
        dir = "towards"
        if car[0] > 4 and car[3] > 0:
            dir = "away from"
        elif car[0] < 4 and car[3] < 0:
            dir = "away from"
        car_states.append(
            (car[1], car[0] - 4, dir, car[2])
        )
    available_actions = []

    if env.move_timer == 0:
        if env.pos > 0:
            available_actions.append("Move down (to Freeway " + str(env.pos - 1) + ")")
        if env.pos < 9:
            available_actions.append("Move up (to Freeway " + str(env.pos + 1) + ")")
    available_actions.append("Stay in the same freeway")
    state_for_llm = {
        'player_states': player_states,
        'car_states': car_states,
        'available_actions': available_actions
    }
    return state_for_llm


class LLMAgent:
    def __init__(self, model, max_new_tokens=2000, token_per_tick=500):
        self.max_new_tokens = max_new_tokens
        self.token_per_tick = token_per_tick
        self.DEBUG = False     
        self.enable_cache = False # True    
        self.write_to_cache = False 
        self.save_trajectory = True # True 
        self.log_csv_dict = {}
        self.model = model 

        self.llm = LLMManager(model_name=self.model, cache_dir=os.getenv('HF_HOME'), max_new_tokens=self.max_new_tokens, token_per_tick=self.token_per_tick)
        
        ### LOGGING ###
        self.time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f'logs/freeway/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)            
        self.log_dir = f'logs/freeway/{model.split('/')[-1]}'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        ### LOGGING ###
        
        self.player_actions = []

        self.llm_system_prompt = "You are a friendly chat assistant who is correct and brief at all times."
        
        self.rules = f'''
1. 10 parallel freeways are numbered from 0 to 9. Freeway 0 is at the bottom and Freeway 9 is at the top. Each can be viewed as an axis in $x$ direction, on which cars travel in $+x$ or $-x$ direction.
2. I start at Freeway 9, and I want to end up in Freeway 0. In each turn, I can move one step, either up or down to the neighbouring freeway or stay.
3. I can only move vertically, which means my x position is fixed to be $x=0$.
4. Each car has different speed in $x$ direction. The speed is given as the number of turns it takes to move 1 unit forward. 
5. I bump into a car if we are in a same freeway and same $x$ position. If that happens, I will be sent back to the starting position on Freeway 9.
6. The episode ends if I get up to Freeway 0.
'''

        # Without COT
        self.base_prompt = f''' I am playing the game """"Freeway"""". Freeway has following rules: {self.rules}. In short, my $x$ position is fixed to be 0. I am at Freeway 9 and want to get to Freeway 0 in minimal number of steps. In this process I need to avoid bumping into cars travelling on the freeways. Help me select my next action and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format.'''
        self.message = [
                    {"role": "system", "content": self.llm_system_prompt},
                    {"role": "user", "content": self.base_prompt},
                ]
            
        self.tokens_used = 0 

        # print(self.all_actions)
        self.log_csv_dict = {}
        self.action_history = []
    def find_best_match(self, action_string):
        if "</think>" not in action_string:
            action_string = STAY_COMPLETION
        else:
            action_string = action_string.split("</think>")[-1]
        match = re.search(r"<answer>(.*?)</answer>", action_string)
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
        #player_states: ([0]: freeway num)
        description = f"I am on Freeway {state_for_llm["player_states"][0]}. {state_for_llm["player_states"][1]}."
        for car in state_for_llm['car_states']:
        #car_states: ([0]: freeway num, [1]: units away, [2]: close to/away from, [3]: speed)
            description += f"There is a car at $x = {car[1]}$ on Freeway {car[0]}. It's moving {car[2]} me and will move 1 unit forward in {car[3]} turns.\n"
        
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
        messages = self.message + [{'role': 'user', 'content': state_description}]
        add_to_dict_list(self.log_csv_dict, 'state_description', state_description)

        action_string = self.llm.inference_fn(messages=messages)
        print(f"{bcolors.OKGREEN}LLM Response: {action_string}{bcolors.ENDC}")
        add_to_dict_list(self.log_csv_dict, 'llm_response', action_string)
        if "</think>" in action_string:
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