import pandas as pd
from utils.client_utils import ApiSingleThreadedLLMClient
from utils.extract_utils import extract_boxed
from vllm import SamplingParams
import time
import re

def meta_controller(args, free, env):
    """
    Meta Controller decides whether to trigger slow agent based on:
    - Environment complexity
    - Slow agent state: whether the agent is still running
    """
    if args.method == "fast":
        return False
    if args.meta_control == "continuous":
        return free
    elif args.meta_control == "periodic":
        return env.env.game_turn % 4 == 0
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
        from envs.overcooked import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.overcooked import SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
    else:
        raise ValueError(f"Game {args.game} is not supported.")
    FORMAT = ACTION_FORMAT_PROMPT if args.format == "A" else CONCLUSION_FORMAT_PROMPT
    
    # set up model client
    client = ApiSingleThreadedLLMClient(args, api_keys)
    # set up env, load prompt - environment specific
    env, real_seed = setup_env(seed, args.difficulty)
    belief_state = ""
    start_time = time.time()
    logs = {
        'description': [], 'render':[], 'meta_control': [],
        'slow_agent_prompt':[], 'slow_agent_response':[], 'belief_state': [],
        'fast_agent_prompt': [], 'fast_agent_response': [], 'follow_plan': [], 
        'action': [], 'reward': [], "slow_response_token_num": [], "fast_response_token_num": []
    }

    while env.env.terminal == False:
        logs['render'].append('\n' + env.env.state_string())
        print("\n" + env.env.state_string())
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm, fast = False)
        fast_agent_response, slow_agent_response = "", ""
        fast_agent_prompt, slow_agent_prompt = "", ""
        fast_response_token_num, slow_response_token_num = 0, 0
        ### --- Slow Agent --- ###
        meta_control = meta_controller(args, client.token_queue_len == 0, env)
        if meta_control:
            messages = [ {"role": "user", "content": SLOW_AGENT_PROMPT + FORMAT + state_description} ]
            slow_agent_prompt = messages[-1]['content']
            if "gemini" in args.slow_model:
                messages[-1]['content'] += "\n Remember to put the action sequence in \\boxed{...} format."
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        slow_agent_response, turns, slow_response_token_num = client.run_slow_inference(messages, "", sampling_params)
        ## --- Update Belief State --- ###
        if args.method == "slow":
            temp = extract_boxed(slow_agent_response)
            if temp != "":
                belief_state = re.sub(r'[^' + ALL_ACTIONS + ']', '', temp)
                belief_state = belief_state[turns:] if len(belief_state) > turns else ""
        elif args.format == "T":
            belief_state = f"""Guidance from a Previous Thinking Model: Turn \( t_1 = {env.env.game_turn - turns} \)\n"""
            belief_state += slow_agent_response
        elif args.format == "A" or args.format == "C":
            if slow_agent_response != "":
                belief_state = f"""**Guidance from a Previous Thinking Model:** Turn \( t_1 = {env.env.game_turn - turns} \)\n"""
                if args.format == "A":
                    belief_state += extract_boxed(slow_agent_response)
                else:
                    belief_state += slow_agent_response.split("</think>")[-1].strip()
        logs['belief_state'].append(belief_state)
        ### --- Fast Agent --- ###
        if args.method == "slow":
            action = belief_state[0] if belief_state != "" else DEFAULT_ACTION
            belief_state = belief_state[1:] if belief_state != "" else ""
        else:
            state_description = state_to_description(state_for_llm, belief_state if belief_state != "" else None, fast = True)
            messages = [ {"role": "user", "content": FAST_AGENT_PROMPT + state_description} ]
            fast_agent_prompt = messages[-1]['content']
            sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=8192)
            fast_agent_response, fast_response_token_num = client.run_fast_inference(messages, sampling_params)
            action = extract_boxed(fast_agent_response)
            action = re.sub(r'[^' + ALL_ACTIONS + ']', '', action)
            if action == "":
                action = DEFAULT_ACTION
            else:
                action = action[-1]
        ### --- Act in Environment --- ###
        r, terminal = env.act(action)
        ### --- Log Information --- ###
        follow_plan = False
        if belief_state != "":
            advice = belief_state.split(f"Turn {env.env.game_turn}: ")[-1].strip()
            advice = re.sub(r'[^' + ALL_ACTIONS + ']', '', advice)
            if advice in ALL_ACTIONS and advice != "":
                follow_plan = advice[0] == action
        logs['description'].append(state_description)
        logs['meta_control'].append(meta_control)
        logs['slow_agent_prompt'].append(slow_agent_prompt)
        logs['slow_agent_response'].append(slow_agent_response)
        logs['fast_agent_prompt'].append(fast_agent_prompt)
        logs['fast_agent_response'].append(fast_agent_response)
        logs['follow_plan'].append(follow_plan)
        logs['action'].append(action)
        logs['reward'].append(env.env.reward)
        logs['slow_response_token_num'].append(slow_response_token_num)
        logs['fast_response_token_num'].append(fast_response_token_num)
        df = pd.DataFrame(logs)
        df.to_csv(file)
        if summarize(seed, args.difficulty, env):
            # clear the belief state and stop the slow agent
            belief_state = ""
            while client.token_queue_len > 0:
                client.run_slow_inference([], "", None)
        print("Reward:", env.env.reward)
    
    # df.to_csv(file)
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