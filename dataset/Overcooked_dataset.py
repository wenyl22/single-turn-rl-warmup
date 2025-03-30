import glob
import random
import argparse
import pandas
from typing import List, Dict, Any
from transformers import AutoTokenizer
from envs.prompts.overcooked import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT
from envs.agent.overcooked_action_manger import LLMActionManager
from envs.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from envs.overcooked_ai_py.mdp.actions import Action
from envs.overcooked_ai_py.mdp.env_descriptions import EnvDescriptions
import ast
import os

def generate_dataset_dict_from_trajectory(trajectory: List[Dict[str, Any]], tokenizer: AutoTokenizer, layout: str, reward: int, tagged: bool):
    print("Generating dataset for layout:", layout)
    dataset = []
    mdp = OvercookedGridworld.from_layout_name(layout)
    state = mdp.get_standard_start_state()
    AM = [
        LLMActionManager(mdp, "player_0", layout),
        LLMActionManager(mdp, "player_1", layout),
    ]
    player_names = ["Alice", "Bob"]
    for i, step in enumerate(trajectory):
        joint_action = ast.literal_eval(step["action"])
        joint_action[0] = Action.ALL_ACTIONS[joint_action[0]]
        joint_action[1] = Action.ALL_ACTIONS[joint_action[1]]
        for _ in range(2):
            state_for_llm = AM[_].prepare_next_move(state)
            description = AM[_].llm_agent._state_to_description(state_for_llm, need_history=False)
            available_actions = AM[_].llm_agent._get_available_actions(state_for_llm)
            if not tagged:
                correct_actions = []
                for action in available_actions:
                    if AM[_].make_next_move(state, action)[0] == joint_action[_]:
                        correct_actions.append(action)
            else:
                correct_actions = [ast.literal_eval(step["tagged_action"])[_]]
            messages = [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "system", "content": LLM_BASE_PROMPT.format(player_name=player_names[_], other_player_name=player_names[1-_], envDescription=EnvDescriptions[layout]) + description}
            ]
            if "deepseek" in tokenizer.name_or_path.lower():
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=False)
                prompt += '<|im_start|>assistant\n<think>'
            if joint_action[_] != Action.STAY:
                # filter stay actions
                dataset.append({
                    "prompt": prompt,
                    "solution": correct_actions,
                    "available_actions": available_actions
                })
            # print("Problem:", prompt)
            # print("Action:", Action.ACTION_TO_CHAR[joint_action[_]])
            # print("Solution:", correct_actions)
            # print("Available actions:", available_actions)
        # print("State:", state)
        prev_state = state
        state, infos = mdp.get_state_transition(prev_state, joint_action)
        reward -= sum(infos["sparse_reward_by_agent"])
    if not tagged:
        assert reward == 0
    return dataset

random.seed(42)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args.add_argument("--tagged", action="store_true")
    args = args.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.tagged:
        files = glob.glob("data/Overcooked_tagged/*/*.csv")
    else:
        files = glob.glob("data/Overcooked/*/*.csv")
    data = []
    
    for file in files:
        layout = file.split("/")[-2]
        reward = int(file.split("_")[-1].split(".")[0])
        df = pandas.read_csv(file)
        df_dict_list = []
        for i in range(len(df)):
            df_dict_list.append(df.iloc[i].to_dict())
        data.extend(generate_dataset_dict_from_trajectory(df_dict_list, tokenizer, layout, reward, args.tagged))
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    test_data = data[split:]
    # store data in parquet format
    train_df = pandas.DataFrame(train_data)
    test_df = pandas.DataFrame(test_data)
    store_path = "data/Overcooked" + ("_tagged" if args.tagged else "")
    store_path += "/DeepSeek" if "deepseek" in tokenizer.name_or_path.lower() else "/Qwen"
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    train_df.to_parquet(store_path + "/train.parquet")
    test_df.to_parquet(store_path + "/test.parquet")
