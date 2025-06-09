import pandas as pd
from utils.client_utils import ApiThreadedLLMClient
from utils.extract_utils import extract_boxed, extract_belief_state
from vllm import SamplingParams
import time
import importlib

VLLM_client = None
def setup_thread_VLLM_client(args):
    global VLLM_client
    VLLM_client = ApiThreadedLLMClient(args) 
   
def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client

def main_game_loop(log_file, seed, args, thread_id):
    # import from envs.{args.game}
    if args.game == "freeway":
        from envs.freeway import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.freeway import LLM_SYSTEM_PROMPT, SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS
    elif args.game == "snake":
        from envs.snake import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.snake import LLM_SYSTEM_PROMPT, SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS
    elif args.game == "airraid":
        from envs.airraid import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.airraid import LLM_SYSTEM_PROMPT, SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS
    else:
        raise ValueError(f"Game {args.game} is not supported.")
        
    # set up model client
    client = VLLM_client
    assert client is not None, "VLLM client is not initialized. Call setup_thread_VLLM_client first."
    client.add_new_thread(thread_id)

    # set up env, load prompt - environment specific
    env = setup_env(seed, args.difficulty)
    belief_state = ""
    start_time = time.time()
    logs = {'description': [], 'render':[], 'fast_agent_prompt': [], 'fast_agent_response': [], 'slow_agent_prompt':[], 'slow_agent_response':[], 'belief_state': [], 'follow_plan': [], 'selected_action': [], 'reward': []}
    # run game loop
    while True:
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm)
        logs['description'].append(state_description)
        logs['render'].append('\n' + env.env.state_string())
        log_fast_agent_response, log_follow_plan, log_slow_agent_response, log_selected_action = "", "", "", ""
        log_fast_agent_prompt, log_slow_agent_prompt = "", ""
        ### --- Slow Agent --- ###
        if args.method != "fast":
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": SLOW_AGENT_PROMPT + state_description}
            ]
            log_slow_agent_prompt = "<system>\n" + LLM_SYSTEM_PROMPT + "</system>\n" + "<user>\n" + SLOW_AGENT_PROMPT + state_description + "</user>\n"
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        # messages will be automatically dropped if client hasn't finished processing the previous request
        turns = client.token_queue_len[thread_id] // client.token_per_tick
        log_slow_agent_response = client.run_slow_inference(thread_id, messages, "", sampling_params)
        if log_slow_agent_response != "":
            belief_state = extract_belief_state(log_slow_agent_response, belief_state, valid_actions = ALL_ACTIONS)
            belief_state = belief_state[turns:] if turns < len(belief_state) else ""
        logs['slow_agent_prompt'].append(log_slow_agent_prompt)
        logs['slow_agent_response'].append(log_slow_agent_response)
        if belief_state == "":
            belief_state = DEFAULT_ACTION
        logs['belief_state'].append(belief_state)

        ### --- Fast Agent --- ###
        if args.method == "slow":
            action = belief_state[0]
        else:
            state_description = state_to_description(state_for_llm, belief_state)
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": FAST_AGENT_PROMPT + state_description}
            ]
            log_fast_agent_prompt = "<system>\n" + LLM_SYSTEM_PROMPT + "</system>\n" + "<user>\n" + FAST_AGENT_PROMPT + state_description + "</user>\n"
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
            log_fast_agent_response = client.run_fast_inference(thread_id, messages, sampling_params)
            action = extract_boxed(log_fast_agent_response)
        log_selected_action = action
        if belief_state[0] == action:
            log_follow_plan = "True"
            belief_state = belief_state[1:]
        else:
            log_follow_plan = "False"
            belief_state = ""
        logs["fast_agent_prompt"].append(log_fast_agent_prompt)
        logs['fast_agent_response'].append(log_fast_agent_response)
        logs['follow_plan'].append(log_follow_plan)
        logs['selected_action'].append(log_selected_action)
        r, terminal = env.act(action)
        logs['reward'].append(env.env.reward)
        df = pd.DataFrame(logs)
        df.to_csv(log_file)
        summarize(seed, args.difficulty, thread_id, env, client)
        if terminal:
            break
    return {
        'seed': seed,
        'game_turn': env.env.game_turn, 
        'reward': env.env.reward,
        'game_time': time.time() - start_time
    }

        
        
        
        
    
        
        
        

