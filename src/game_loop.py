import pandas as pd
from utils.client_utils import ApiSingleThreadedLLMClient
from utils.extract_utils import extract_boxed
from vllm import SamplingParams
import time
import re
import os
from utils.exe_utils import execute_code

def meta_controller(args, idle, env):
    """
    Meta Controller decides whether to trigger slow agent based on:
    - Environment complexity
    - Slow agent state: whether the agent is still running
    """
    if args.method == "fast":
        return False
    if args.meta_control == "continuous":
        return idle
    elif args.meta_control.startswith("periodic"):
        f = int(args.meta_control[8:])
        return env.env.game_turn % f == 0
    elif args.meta_control == "triggered":
        pass

def main_game_loop(file, seed, args, api_keys):
    # import from envs.{args.game}
    if args.game == "freeway":
        from envs.freeway import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.freeway import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
    elif args.game == "snake":
        from envs.snake import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.snake import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
    elif args.game == "airraid":
        from envs.airraid import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.airraid import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
    elif args.game == "overcooked":
        from envs.overcooked import setup_env, llm_state_builder, state_to_description, state_to_json, summarize
        from envs.prompts.overcooked import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT, CODE_GENERATOR_PROMPT
    else:
        raise ValueError(f"Game {args.game} is not supported.")
    FORMAT = ACTION_FORMAT_PROMPT if args.format == "A" else CONCLUSION_FORMAT_PROMPT
    
    # set up model client
    client = ApiSingleThreadedLLMClient(args, api_keys)
    # set up env, load prompt - environment specific
    env, real_seed = setup_env(seed, args.difficulty)
    memory = ""
    start_time = time.time()
    logs = {
        'description': [], 'render':[], 'meta_control': [],
        'slow_agent_prompt':[], 'slow_agent_response':[], 'memory': [],
        'fast_agent_prompt': [], 'fast_agent_response': [], 
        'action': [], 'reward': [], "slow_response_token_num": [], "fast_response_token_num": []
    }
    while env.env.terminal == False:
        logs['render'].append('\n' + env.env.state_string())
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm, fast = False, json_mode=args.method == "codegen")
        fast_agent_response, slow_agent_response = "", ""
        fast_agent_prompt, slow_agent_prompt = "", ""
        fast_response_token_num, slow_response_token_num = 0, 0
        ### --- Slow Agent --- ###
        meta_control = meta_controller(args, client.gen_text == "", env)
        if meta_control:
            messages = [ {"role": "user", "content": SLOW_AGENT_PROMPT + FORMAT + state_description} ]
            if args.method == "codegen":
                messages = [ {"role": "user", "content": CODE_GENERATOR_PROMPT + FORMAT + state_description} ]
            slow_agent_prompt = messages[-1]['content']
            if "gemini" in args.slow_model:
                messages[-1]['content'] += "\n Remember to put the action sequence in \\boxed{...} format."
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        slow_agent_response, turns, slow_response_token_num = client.run_slow_inference(messages, sampling_params, env.env.game_turn)
        ## --- Update Persistent Memory --- ###
        if args.method == "slow":
            temp = extract_boxed(slow_agent_response)
            memory = re.sub(r'[^' + ALL_ACTIONS + ']', '', temp)
            memory = memory[turns:] if len(memory) > turns else ""
        if slow_agent_response != "":
            memory = f"""**Guidance from a Previous Thinking Model:** Turn \( t_1 = {turns} \)\n"""
            if args.format == "A":
                memory += extract_boxed(slow_agent_response)
            elif args.format == "C":
                memory += slow_agent_response.split("</think>")[-1].strip()
            else:
                memory += slow_agent_response
        logs['memory'].append(memory)
        ### --- Fast Agent --- ###
        if args.method == "codegen":
            python_code_str = slow_agent_response.split('```python')[-1].split('```')[0].strip()
            json_state = state_to_json(state_for_llm)
            success, result = execute_code(python_code_str, json_state, timeout=60)
            if not success or not result in ALL_ACTIONS:
                action = DEFAULT_ACTION
                fast_agent_response = f"Error executing code: {result}"
            else: 
                fast_agent_response = result
                action = result
        elif args.method == "slow":
            action = memory[0] if memory != "" else DEFAULT_ACTION
            memory = memory[1:] if memory != "" else ""
        else:
            state_description = state_to_description(state_for_llm, memory if memory != "" else None, fast=True)
            messages = [ {"role": "user", "content": FAST_AGENT_PROMPT + state_description} ]
            fast_agent_prompt = messages[-1]['content']
            sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=args.fast_max_token)
            fast_agent_response, fast_response_token_num = client.run_fast_inference(messages, sampling_params, ALL_ACTIONS, DEFAULT_ACTION)
            action = extract_boxed(fast_agent_response)
        ### --- Act in Environment --- ###
        r, terminal = env.act(action)
        ### --- Log Information --- ###
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
        df = pd.DataFrame(logs)
        df.to_csv(file)
        if summarize(seed, args.difficulty, env):
            # clear the belief state and stop the slow agent
            memory = ""
            while client.gen_text != "":
                client.run_slow_inference([], None, None)
            client.to_flush = ""
    df = pd.DataFrame(logs)
    df.to_csv(file)
    dir = '/'.join(file.split('/')[:-1])
    slow_response_token = [_ for _ in logs["slow_response_token_num"] if _ > 0]
    fast_response_token = [_ for _ in logs["fast_response_token_num"] if _ > 0]
    mean_slow_response_token_num = sum(slow_response_token) / len(slow_response_token) if len(slow_response_token) > 0 else 0
    fast_fast_response_token_num = sum(fast_response_token) / len(fast_response_token) if len(fast_response_token) > 0 else 0
    return {
        'logdir': dir,
        'seed': real_seed,
        'game_turn': env.env.game_turn, 
        'reward': env.env.reward,
        'game_time': time.time() - start_time,
        'slow_response_token_num': mean_slow_response_token_num,
        'fast_response_token_num': fast_fast_response_token_num,
    }