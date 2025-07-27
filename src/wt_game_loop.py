import time
import queue
import threading
from utils.extract_utils import extract_boxed
from vllm import SamplingParams
from utils.client_utils import WallTimeLLMClients
from utils.display_utils import Logger
import os
import pandas as pd
import re
class GamePlay():
    def __init__(self):
        # Env - > slow
        # 1. message: empty means not invoked
        # 2. poll: get memory

        # Env - > fast
        # 1. message: always non-empty
        # 2. poll: get action
        
        # fast, slow - > Env
        # 1. memory: from slow agent
        # 2. action: from fast agent
        self.fast2env = queue.Queue()
        self.slow2env = queue.Queue()
        # fast control
        self.fast_thread = None
        self.fast_content = ""
        self.stop_fast = False
        
        # slow control
        self.slow_thread = None
        self.slow_content = ""
        self.stop_slow = False
    def meta_controller(self, args, env):
        if args.method == "fast":
            return False
        if args.meta_control == "continuous":
            return self.slow_thread is None or not self.slow_thread.is_alive()
        if args.meta_control.startswith("periodic"):
            f = int(args.meta_control[8:])
            return env.env.game_turn % f == 0
    def handle_Q2fast(self, event):
        if event["type"] == "message":
            if self.fast_thread and self.fast_thread.is_alive():
                self.stop_fast = True
                self.fast_thread.join(timeout=3.0)
            self.stop_fast = False
            self.fast_content = ""
            self.fast_thread = threading.Thread(target=self.run_fast_inference, args=(event["content"],), daemon=True)
            self.fast_thread.start()
        elif event["type"] == "poll":
            if self.fast_content != "":
                self.fast2env.put({"type": "fast", "content": self.fast_content})
            time.sleep(0.1)            
    def handle_Q2slow(self, event):
        if event["type"] == "message":
            if self.slow_thread and self.slow_thread.is_alive():
                self.stop_slow = True
                self.slow_thread.join(timeout=3.0)
            self.stop_slow = False
            self.slow_content = ""
            self.slow_turn = event["turn"]
            self.slow_thread = threading.Thread(target=self.run_slow_inference, args=(event["content"],), daemon=True)
            self.slow_thread.start()
        elif event["type"] == "poll":
            if self.format == "T":
                self.slow2env.put({"type": "slow", "content": self.slow_content, "turn": self.slow_turn})
            time.sleep(0.1)
    def handle_slow2env(self):
        _s, _turn = "", 0
        while not self.slow2env.empty():
            event = self.slow2env.get()
            _s = event["content"]
            _turn = event["turn"]
        return _s, _turn
    def handle_fast2env(self):
        _f = ""
        while not self.fast2env.empty():
            event = self.fast2env.get()
            _f = event["content"]
        return _f
    def run_env(self, log_dir, seed, args):
        if args.game == "freeway":
            from envs.freeway import setup_env, llm_state_builder, state_to_description, summarize
            from envs.prompts.freeway import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
        elif args.game == "snake":
            from envs.snake import setup_env, llm_state_builder, state_to_description, summarize
            from envs.prompts.snake import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
        elif args.game == "overcooked":
            from envs.overcooked import setup_env, llm_state_builder, state_to_description, summarize
            from envs.prompts.overcooked import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
        else:
            raise ValueError(f"Game {args.game} is not supported.")
        self.fast_agent_token = int(args.fast_agent_time / 0.045)
        self.format = args.format
        FORMAT = ACTION_FORMAT_PROMPT if args.format == "A" else CONCLUSION_FORMAT_PROMPT
        env, real_seed = setup_env(seed, args.difficulty)
        self.client = WallTimeLLMClients(args, args.api_keys)        
        memory = ""
        last_t = time.time()
        self.logger = Logger(args, log_dir, seed, real_seed)
        logs = {
            'description': [], 'render':[], 'meta_control': [],
            'slow_agent_prompt':[], 'slow_agent_response':[], 'memory': [],
            'fast_agent_prompt': [], 'fast_agent_response': [], 
            'action': [], 'reward': [], "slow_response_token_num": [], "fast_response_token_num": []
        }
        while not env.env.terminal:
            logs['render'].append('\n' + env.env.state_string())
            fast_agent_response, slow_agent_response = "", ""
            fast_agent_prompt, slow_agent_prompt = "", ""
            fast_response_token_num, slow_response_token_num = 0, 0
            state_for_llm = llm_state_builder(env.env)
            state_description = state_to_description(state_for_llm, fast = False)
            meta_control = self.meta_controller(args, env)
            ### --- Logging --- ###
            self.logger.log_turn_start(env, meta_control)
            ### --- Logging --- ###

            if meta_control:
                slow_agent_prompt = SLOW_AGENT_PROMPT + FORMAT + state_description
                self.handle_Q2slow({"type": "message", "content": SLOW_AGENT_PROMPT + FORMAT + state_description, "turn": env.env.game_turn})
            time.sleep(args.seconds_per_step - args.fast_agent_time) # idle fast agent, running slow agent

            self.handle_Q2slow({"type": "poll"}) # poll for memory
            _s, _turn = self.handle_slow2env()
            slow_agent_response = _s
            slow_response_token_num = int(0.3 * len(slow_agent_response))
            if args.method == "slow":
                temp = extract_boxed(_s)
                memory = re.sub(r'[^' + ALL_ACTIONS + ']', '', temp)
                if args.game != 'overcooked':
                    memory = memory[env.env.game_turn - _turn:] if len(memory) > env.env.game_turn - _turn else ""
            elif _s is not None  and _s != "":
                memory = f"""**Guidance from a Previous Thinking Model:** Turn \( t_1 = {_turn} \)\n"""
                if args.format == "A":
                    memory += extract_boxed(_s)
                elif args.format == "C":
                    memory += _s.split("</think>")[-1].strip()
                else:
                    memory += _s
            logs['memory'].append(memory)

            if args.method != "slow":
                state_description = state_to_description(state_for_llm, memory if memory != "" else None, fast = True)
                fast_agent_prompt = FAST_AGENT_PROMPT + state_description
                self.handle_Q2fast({"type": "message", "content": FAST_AGENT_PROMPT + state_description})
            time.sleep(max(args.seconds_per_step - (time.time() - last_t), 0)) # parallel running fast & slow

            last_t = time.time()
            self.handle_Q2fast({"type": "poll"}) # poll for action
            _f = self.handle_fast2env()
            fast_agent_response = _f
            fast_response_token_num = int(0.3 * len(fast_agent_response))
            if args.method == "slow":
                action = memory[0] if memory != "" else DEFAULT_ACTION
                memory = memory[1:] if memory != "" else ""
            else:
                action = extract_boxed(_f) if _f != "" else DEFAULT_ACTION
                action = re.sub(r'[^' + ALL_ACTIONS + ']', '', action)
                if action == "":
                    action = DEFAULT_ACTION
                else:
                    action = action[0]
            r, t = env.act(action)
            ### --- Logging --- ###
            self.logger.log_action_execution(env, action)
            logs['description'].append(state_description)
            logs['meta_control'].append(meta_control)
            logs['slow_agent_prompt'].append(slow_agent_prompt)
            logs['slow_agent_response'].append(slow_agent_response)
            logs['fast_agent_prompt'].append(fast_agent_prompt)
            logs['fast_agent_response'].append(fast_agent_response)
            logs['action'].append(action)
            logs['reward'].append(env.env.reward)
            logs['slow_response_token_num'].append(slow_response_token_num)
            logs['fast_response_token_num'].append(fast_response_token_num)
            ### --- Logging --- ###
            if summarize(seed, args.difficulty, env):
                if self.slow_thread and self.slow_thread.is_alive():
                    self.stop_slow = True
                    self.slow_thread.join(timeout=3.0)
                if self.fast_thread and self.fast_thread.is_alive():
                    self.stop_fast = True
                    self.fast_thread.join(timeout=3.0)
                memory = ""
            df = pd.DataFrame(logs)
            df.to_csv(f"{log_dir}/{seed}.csv")
        self.logger.save_final_logs()
        print(f"Game {args.game} with seed {real_seed} finished. Total turns: {env.env.game_turn}, Reward: {env.env.reward}")
        return {
            "logdir": log_dir,
            "seed": real_seed,
            "time": time.time() - self.logger.game_start_time,
            "turns": env.env.game_turn,
            "reward": env.env.reward,
        }

    def run_fast_inference(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=self.fast_agent_token)
        response = self.client.generate(messages, sampling_params, fast=True)
        self.fast_content = ""
        for chunk in response:
            generation = ""
            if chunk.choices[0].delta.content:
                generation += chunk.choices[0].delta.content
            if generation != "":
                self.logger.log_streaming_content("fast", generation, self.fast_content != "")
                self.fast_content += generation
            if self.stop_fast:
                break
        if "oxed" in self.fast_content:
            self.fast2env.put({"type": "fast", "content": self.fast_content})
            return
        generation = "\n Therefore, the final answer is \\boxed{"
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": self.fast_content + generation}]
        max_attempt = 3
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1)
        generation += ""
        while max_attempt > 0:
            temp = self.client.fast_llm.chat.completions.create(
                model=self.client.fast_model, messages=messages, max_tokens=1, temperature=0, top_p=1,
            )
            if temp.choices[0].message.content.strip()[0] in ['U', 'D', 'L', 'R', 'S', 'I']:
                generation += temp.choices[0].message.content.strip()[0] + '}'
                break
            max_attempt -= 1
        self.fast_content += generation
        self.logger.log_streaming_content("fast", generation, self.fast_content != "")
        self.fast2env.put({"type": "fast", "content": self.fast_content})
        return
    def run_slow_inference(self, prompt):
        reasoning = False
        messages = [{"role": "user", "content": prompt}]
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        response = self.client.generate(messages, sampling_params, fast=False)
        self.slow_content = ""
        for chunk in response:
            generation= ""
            if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content is not None:
                if reasoning == False:
                    reasoning = True
                    generation += '<think>'
                generation += chunk.choices[0].delta.reasoning_content
            if chunk.choices[0].delta.content:
                if reasoning == True:
                    reasoning = False
                    generation += '</think>\n'
                generation += chunk.choices[0].delta.content
            if generation != "":
                self.logger.log_streaming_content("slow", generation, self.slow_content != "")
                self.slow_content += generation
            if self.stop_slow:
                break
        self.slow2env.put({"type": "slow", "content": self.slow_content, "turn": self.slow_turn})
        return
