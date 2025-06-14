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

def main_game_loop(file, ckpt, seed, args, thread_id):
    # import from envs.{args.game}
    if args.game == "freeway":
        from envs.freeway import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.freeway import LLM_SYSTEM_PROMPT, SLOW_AGENT_PROMPT, FAST_AGENT_ACTION_PROMPT, FAST_AGENT_CONCLUSION_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
    elif args.game == "snake":
        from envs.snake import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.snake import LLM_SYSTEM_PROMPT, SLOW_AGENT_PROMPT, FAST_AGENT_ACTION_PROMPT, FAST_AGENT_CONCLUSION_PROMPT, DEFAULT_ACTION, ALL_ACTIONS, ACTION_FORMAT_PROMPT, CONCLUSION_FORMAT_PROMPT
    elif args.game == "airraid":
        from envs.airraid import setup_env, llm_state_builder, state_to_description, summarize
        from envs.prompts.airraid import LLM_SYSTEM_PROMPT, SLOW_AGENT_PROMPT, FAST_AGENT_PROMPT, DEFAULT_ACTION, ALL_ACTIONS
    else:
        raise ValueError(f"Game {args.game} is not supported.")
    FORMAT = ACTION_FORMAT_PROMPT if args.format == "A" else CONCLUSION_FORMAT_PROMPT
    FAST_AGENT_PROMPT = FAST_AGENT_ACTION_PROMPT if args.format == "A" else FAST_AGENT_CONCLUSION_PROMPT
    
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
    
    # load checkpoint if exists
    if ckpt is not None:
        logs = pd.read_csv(ckpt, index_col=0).to_dict(orient='list')
        for a in logs['action']:
            r, t = env.act(a)
            if r < 0 and not t:
                game_turn = env.env.game_turn
                reward = env.env.reward
                env.seed(real_seed[0])
                env.reset()
                env.env.game_turn = game_turn
                env.env.reward = reward
        # print(f"Thread {thread_id} loaded checkpoint from {ckpt} with {len(logs['action'])} actions.")
    while env.env.terminal == False:
        logs['render'].append('\n' + env.env.state_string())
        state_for_llm = llm_state_builder(env.env)
        state_description = state_to_description(state_for_llm)
        fast_agent_response, slow_agent_response = "", ""
        fast_agent_prompt, slow_agent_prompt = "", ""
        ### --- Slow Agent --- ###
        meta_control = meta_controller(args, client.token_queue_len[thread_id] == 0, env)
        if meta_control:
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": SLOW_AGENT_PROMPT + FORMAT + state_description}
            ]
            slow_agent_prompt = messages[-1]['content']
        else:
            messages = []
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=32768)
        slow_agent_response, turns = client.run_slow_inference(thread_id, messages, "", sampling_params)
        ### --- Update Belief State --- ###
        if slow_agent_response.split("</think>")[-1] != "":
            belief_state = f"""**Guidance from a Previous Thinking Model \(Turn {env.env.game_turn - turns}\)**\n"""
            if args.format == "A":
                belief_state += extract_boxed(slow_agent_response)
            else:
                belief_state += slow_agent_response.split("</think>")[-1].strip()
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
            fast_agent_prompt = messages[-1]['content']
            sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
            fast_agent_response = client.run_fast_inference(thread_id, messages, sampling_params)
            action = extract_boxed(fast_agent_response)
        ### --- Act in Environment --- ###
        r, terminal = env.act(action)
        ### --- Log Information --- ###
        belief_state_action = belief_state.split(f"Turn {env.env.game_turn + 1}: ")[-1].split('\n')[0].strip()
        follow_plan = True
        if belief_state_action in ALL_ACTIONS and belief_state_action != "":
            follow_plan = belief_state_action == action
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
        if summarize(seed, args.difficulty, thread_id, env):
            # clear the belief state and stop the slow agent
            belief_state = ""
            while client.token_queue_len[thread_id] > 0:
                client.run_slow_inference(thread_id, [], "", None)
    
    df.to_csv(file)
    return {
        'seed': real_seed,
        'game_turn': env.env.game_turn, 
        'reward': env.env.reward,
        'game_time': time.time() - start_time
    }