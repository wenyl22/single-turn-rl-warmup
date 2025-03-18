import sys
import os

from Overcooked_prompts import LLM_SYSTEM_PROMPT, BASE_PROMPT
from overcooked_ai_py.mdp.env_descriptions import EnvDescriptions
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from Overcooked_action_manger import LLMActionManager
from overcooked_ai_py.mdp.actions import Action, Direction
import time 
import numpy as np 
from tqdm import tqdm 
import argparse 
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
from fuzzywuzzy import process
import re
import pandas as pd
import datetime
def find_best_match(action_string, available_actions_list):
    action_string = action_string.split("</think>")[-1]
    # if action string does not contain any english character, it's "wait."
    if not any(char.isalpha() for char in action_string):
        return "wait."
    match = re.search(r"<answer>(.*?)</answer>", action_string)
    if match:
        selected_match = match.group(1).strip()
    else:
        selected_match = action_string
    for action in available_actions_list:
        if selected_match.lower() in action.lower():
            return action 
    selected_move, score = process.extractOne(selected_match, available_actions_list)
    return selected_move
import random
global_llm = None
global_tokenizer = None
def main(layout, model_name, max_new_tokens, token_per_tick):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    if global_llm is None:
        global_tokenizer = AutoTokenizer.from_pretrained(model_name)
        global_model = LLM(model_name)
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=max_new_tokens,
        top_p=0.9
    )
    mdp = OvercookedGridworld.from_layout_name(layout)
    state = mdp.get_standard_start_state()
    AM = [
        LLMActionManager(mdp, "player_0", layout),
        LLMActionManager(mdp, "player_1", layout),
    ]
    player_names = ["Alice", "Bob"]
    score = 0
    NUM_TICKS = 400
    reward_freq = 0
    logs = [
        {
            "description":[],
            "llm_response":[],
            "selected_action":[],
            "reward":[]
        },
        {
            "description":[],
            "llm_response":[],
            "selected_action":[],
            "reward":[]
        }
    ]
    for tick in tqdm(range(NUM_TICKS)):
        joint_action = [Action.STAY] * 2
        for _ in range(2):
            state_for_llm = AM[_].prepare_next_move(state)
            description = AM[_].llm_agent._state_to_description(state_for_llm, need_history=True)
            available_actions = AM[_].llm_agent._get_available_actions(state_for_llm, '')
            MESSAGE = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "system", "content": BASE_PROMPT.format(player_name=player_names[_], other_player_name=player_names[1-_], envDescription=EnvDescriptions[layout])}
            ]
            prompt = MESSAGE + [{"role": "user", "content": description}]
            if "deepseek" in global_tokenizer.name_or_path.lower():
                prompt = global_tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            else:
                prompt = global_tokenizer.apply_chat_template(prompt, add_special_tokens=False, tokenize=False)
                prompt += '<|im_start|>assistant\n<think>'
            response = global_model.generate(prompt, sampling_params)[0].outputs[0].text
            if "</think>" in response:
                selected_action = find_best_match(response, available_actions)
            else:
                selected_action = "wait."
            joint_action[_] = AM[_].make_next_move(state, selected_action)
            logs[_]["description"].append(description)
            logs[_]["llm_response"].append(response)
            logs[_]["selected_action"].append(selected_action)
        prev_state = state
        state, infos = mdp.get_state_transition(prev_state, joint_action)
        score += sum(infos["sparse_reward_by_agent"])
        logs[0]["reward"].append(infos["shaped_reward_by_agent"][0])
        logs[1]["reward"].append(infos["shaped_reward_by_agent"][1])
        if infos["shaped_reward_by_agent"][0] > 0:
            reward_freq += 1
        if infos["shaped_reward_by_agent"][1] > 0:
            reward_freq += 1
        df1 = pd.DataFrame(logs[0])
        df1.to_csv(f'game_logs/overcooked-ai/{layout}/{model_name.split("/")[-1]}/{player_names[0]}_{max_new_tokens}_{token_per_tick}_{time_stamp}.csv')
        df2 = pd.DataFrame(logs[1])
        df2.to_csv(f'game_logs/overcooked-ai/{layout}/{model_name.split("/")[-1]}/{player_names[1]}_{max_new_tokens}_{token_per_tick}_{time_stamp}.csv')
        #print(logs)
        #exit(0)
        print("Tick:", tick, "Score:", score, "Reward Freq:", reward_freq)
        print("----------------------------------------------------------")
        #exit(0)
    return score, reward_freq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Overcooked benchmark with a specific model.')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', help='Model name to benchmark')
    parser.add_argument('--max_new_tokens', type=int, default=2000)
    parser.add_argument('--token_per_tick', type=int, default=500)
    parser.add_argument('--layout_name', choices=['forced_coordination', 'cramped_room', 'counter_circuit_o_1order', 'asymmetric_advantages', 'coordination_ring'], default='forced_coordination')
    args = parser.parse_args()

    model_name = args.model_name
    max_new_tokens = args.max_new_tokens
    token_per_tick = args.token_per_tick
    layout_name = args.layout_name
    print(f'Benchmarking model: {model_name}')
    NUM_TRIALS = 3
    scores = []
    game_times = []
    reward_freqs = []
    if not os.path.exists(f'game_logs/overcooked-ai/{layout_name}'):
        os.makedirs(f'game_logs/overcooked-ai/{layout_name}')
    if not os.path.exists(f'game_logs/overcooked-ai/{layout_name}/{model_name.split("/")[-1]}'):
        os.makedirs(f'game_logs/overcooked-ai/{layout_name}/{model_name.split("/")[-1]}')

    for idx in range(NUM_TRIALS):
        start_time = time.time()
        score, reward_freq = main(layout_name, model_name, max_new_tokens, token_per_tick)
        scores.append(score)
        reward_freqs.append(reward_freq)
        game_time = time.time() - start_time
        game_times.append(game_time)
    with open(f'game_logs/overcooked-ai/{layout_name}/{model_name.split('/')[-1]}/summarize_{max_new_tokens}_{token_per_tick}.txt', 'w') as f:
        f.write(f"MEAN SCORE: {np.mean(scores)}\n")
        f.write(f"STD ERROR: {np.std(np.array(scores)) / np.sqrt(NUM_TRIALS)}\n")
        f.write(f"GAME SCORES: {scores}\n")
        f.write(f"MEAN GAME TIME: {np.mean(game_times)}\n")
        f.write(f"GAME TIMES: {game_times}\n")
        f.write(f"MEAN REWARD FREQ: {np.mean(reward_freqs)}\n")
        f.write(f"GAME REWARD FREQS: {reward_freqs}\n")