
from envs.overcooked_ai_py.mdp.env_descriptions import EnvDescriptions
from envs.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from envs.agent.overcooked_action_manger import LLMActionManager
from envs.overcooked_ai_py.mdp.actions import Action
from envs.prompts.overcooked import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT, STAY_COMPLETION
from envs.utils import find_best_match, LocalThreadedLLMClient
import time 
import numpy as np 
from tqdm import tqdm 
import pandas as pd
import random
VLLM_client = None 

def setup_thread_VLLM_client(token_per_tick):
    global VLLM_client
    VLLM_client = LocalThreadedLLMClient(token_per_tick=token_per_tick)

def get_thread_VLLM_client():
    global VLLM_client
    return VLLM_client
def overcooked_game_loop(log_file, seed, difficulty = 1):
    client = VLLM_client
    assert client is not None, "VLLM client is not initialized. Please call setup_thread_VLLM_client() first."
    thread_id = client.add_new_thread()
    log_files = [log_file.replace(".csv", "Alice.csv"), log_file.replace(".csv", "Bob.csv")]
    
    random.seed(seed)
    np.random.seed(seed)
    layout = "asymmetric_advantages"
    mdp = OvercookedGridworld.from_layout_name(layout)
    state = mdp.get_standard_start_state()
    player_names = ["Alice", "Bob"]
    AM = [LLMActionManager(mdp, f"player_{_}", layout) for _ in range(2)]
    logs = [{"description":[], "llm_response":[], "selected_action":[], "reward":[]} for _ in range(2)]
    NUM_TICKS = 400
    score = 0
    shaped_score = 0
    start_time = time.time()
    for tick in range(NUM_TICKS):
        joint_action = [Action.STAY] * 2
        for _ in range(2):
            state_for_llm = AM[_].prepare_next_move(state)
            description = AM[_].llm_agent._state_to_description(state_for_llm, need_history=True)
            available_actions = AM[_].llm_agent._get_available_actions(state_for_llm)
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": LLM_BASE_PROMPT.format(env_description=EnvDescriptions[layout]) + description}
            ]
            response = client.run_inference(thread_id, messages, STAY_COMPLETION)
            selected_action = find_best_match(response, available_actions, STAY_COMPLETION)
            joint_action[_] = AM[_].make_next_move(state, selected_action)
            logs[_]["description"].append(description)
            logs[_]["llm_response"].append(response)
            logs[_]["selected_action"].append(selected_action)
        prev_state = state
        state, infos = mdp.get_state_transition(prev_state, joint_action)
        score += sum(infos["sparse_reward_by_agent"])
        shaped_score += sum(infos["shaped_reward_by_agent"])
        for _ in range(2):
            logs[_]["reward"].append(infos["sparse_reward_by_agent"][_])
            df = pd.DataFrame(logs[_])
            df.to_csv(log_files[_], index=False)
        print("Tick:", tick, "Score:", score, "Shaped Score:", shaped_score)
    return {
        "seed": seed,
        "score": score,
        "shaped_score": shaped_score,
        "game_time": time.time() - start_time,
    }