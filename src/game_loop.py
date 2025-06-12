import pandas as pd
from utils.client_utils import ApiThreadedLLMClient
from utils.extract_utils import extract_boxed, extract_belief_state
from vllm import SamplingParams
import time

VLLM_client = None
def setup_thread_VLLM_client(args):
    global VLLM_client
    VLLM_client = ApiThreadedLLMClient(args) 
   
def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client

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

def update_belief_state(args, text, old_belief_state, valid_actions, passed_turns):
    """
    Update belief state based on the slow agent response.
    """
    if args.format == "A":
        belief_state = extract_belief_state(text, old_belief_state, valid_actions=valid_actions)
        belief_state = belief_state[passed_turns:] if passed_turns < len(belief_state) else ""
        return belief_state
    elif args.format == "AC":
        return extract_belief_state(text, belief_state, valid_actions=valid_actions, with_conclusion=True)
    elif args.format == "TA":
        return extract_belief_state(text, belief_state, valid_actions=valid_actions, with_thinking=True)
    else:
        raise ValueError(f"Format {args.format} is not supported.")

def main_game_loop(file, seed, args, thread_id):
    # import from envs.{args.game}
    if args.game == "freeway":
        from envs.freeway import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.freeway import LLM_SYSTEM_PROMPT, SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, SEQUENCE_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
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
    env, real_seed = setup_env(seed, args.difficulty)
    belief_state = ""
    start_time = time.time()
    logs = {
        'description': [], 'render':[], 'meta_control': [],
        'slow_agent_prompt':[], 'slow_agent_response':[], 'belief_state': [],
        'fast_agent_prompt': [], 'fast_agent_response': [], 'follow_plan': [], 
        'action': [], 'reward': []
    }
    while True:
        logs['render'].append('\n' + env.env.state_string())
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm)
        fast_agent_response, slow_agent_response = "", ""
        fast_agent_prompt, slow_agent_prompt = "", ""
        ### --- Slow Agent --- ###
        meta_control = meta_controller(args, client.token_queue_len[thread_id] == 0, env)
        if meta_control:
            FORMAT = CONCLUSION_FORMAT_PROMPT if args.format == "AC" else SEQUENCE_FORMAT_PROMPT
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": SLOW_AGENT_PROMPT + FORMAT + state_description}
            ]
            slow_agent_prompt = messages[0]['content']
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        slow_agent_response, turns = client.run_slow_inference(thread_id, messages, "", sampling_params)
        ### --- Update Belief State --- ###
        if slow_agent_response != "":
            belief_state = update_belief_state(args, slow_agent_response, belief_state, ALL_ACTIONS, turns)
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
            fast_agent_prompt = messages[0]['content']
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
            fast_agent_response = client.run_fast_inference(thread_id, messages, sampling_params)
            action = extract_boxed(fast_agent_response)
        ### --- Act in Environment --- ###
        r, terminal = env.act(action)
        ### --- Log Information --- ###
        if belief_state[0] == action:
            follow_plan = True
            belief_state = belief_state[1:]
        else:
            follow_plan = False
            belief_state = ""
        logs['description'].append(state_description)
        logs['meta_control'].append(meta_control)
        logs['slow_agent_prompt'].append(slow_agent_prompt)
        logs['slow_agent_response'].append(slow_agent_response)
        logs["fast_agent_prompt"].append(fast_agent_prompt)
        logs['fast_agent_response'].append(fast_agent_response)
        logs['follow_plan'].append(follow_plan)
        logs['action'].append(action)
        logs['reward'].append(env.env.reward)
        df = pd.DataFrame(logs)
        df.to_csv(file)
        summarize(seed, args.difficulty, thread_id, env, client)
        if terminal:
            break
    return {
        'seed': real_seed,
        'game_turn': env.env.game_turn, 
        'reward': env.env.reward,
        'game_time': time.time() - start_time
    }