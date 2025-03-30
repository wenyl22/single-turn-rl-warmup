import glob
import random
import argparse
import pandas
from typing import List, Dict, Any
from transformers import AutoTokenizer
from envs.prompts.overcooked import LLM_SYSTEM_PROMPT, BASE_PROMPT
from envs.overcooked_action_manger import LLMActionManager, extract_location
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.env_descriptions import EnvDescriptions
import ast
import os
import pandas as pd

Terrain = {
    "X": "counter",
    "S": "soup",
    "P": "cooker",
    "O": "onion",
    "D": "plate"
}

def generate_dataset_dict_from_trajectory(trajectory: List[Dict[str, Any]], tokenizer: AutoTokenizer, layout: str, reward: int, id: int):
    print("Generating dataset for layout:", layout)
    dataset = []
    mdp = OvercookedGridworld.from_layout_name(layout)
    state = mdp.get_standard_start_state()
    AM = [
        LLMActionManager(mdp, "player_0", layout),
        LLMActionManager(mdp, "player_1", layout),
    ]
    player_names = ["Alice", "Bob"]
    last_interaction = [-1, -1]
    tagged_actions = [[], []]
    state_graph = []
    score = reward
    for i, step in enumerate(trajectory):
        joint_action = ast.literal_eval(step["action"])
        joint_action[0] = Action.ALL_ACTIONS[joint_action[0]]
        joint_action[1] = Action.ALL_ACTIONS[joint_action[1]]
        new_state = state.deepcopy()
        for _ in range(2):
            if joint_action[_] != Action.INTERACT:
                continue
            # check what it is interacting with
            pos, o = new_state.players[_].position, new_state.players[_].orientation
            i_pos = Action.move_in_direction(pos, o)
            target_terrain_type = mdp.get_terrain_type_at_pos(i_pos)
            # terrain type: X, S, P, O, D (counter, soup, pot, onion, dish)
            state_for_llm = AM[_].prepare_next_move(state)
            description = AM[_].llm_agent._state_to_description(state_for_llm, False)
            available_actions = AM[_].llm_agent._get_available_actions(state_for_llm)
            possible_targets = []
            for action in available_actions:
                if "wait" in action:
                    target = action
                    continue
                elif "away" in action:
                    continue
                location, location_id = extract_location(action)
                #print(available_actions, action, location, list(AM[_].object_location.keys()), location in list(AM[_].object_location.keys()))
                assert location in list(AM[_].object_location.keys())
                location_coords = AM[_].object_location[location][int(location_id)]
                if location_coords != i_pos:
                    continue
                possible_targets.append(action)            
            if len(possible_targets) == 0:
                pass # wait
            elif len(possible_targets) == 1:
                target = possible_targets[0]
            else:
                raise ValueError("Multiple possible targets")
            tagged_actions[_].extend([target] * (i - last_interaction[_]))
            last_interaction[_] = i
        state_graph.append(mdp.state_string(state))
        prev_state = state
        state, infos = mdp.get_state_transition(prev_state, joint_action)
        reward -= sum(infos["sparse_reward_by_agent"])
    if not os.path.exists(f"data/Overcooked_tagged/{layout}"):
        os.makedirs(f"data/Overcooked_tagged/{layout}")
    tagged_trajectory = []
    for i in range(min(len(tagged_actions[0]), len(tagged_actions[1]))):
        tagged_trajectory.append({
            "state": trajectory[i]["state"],
            "sparse_reward": trajectory[i]["sparse_reward"],
            "shaped_reward": trajectory[i]["shaped_reward"],
            "tagged_action": [tagged_actions[0][i], tagged_actions[1][i]],
            "action": trajectory[i]["action"]
        })
    pd.DataFrame(tagged_trajectory).to_csv(f"data/Overcooked_tagged/{layout}/{id}_{score}.csv")
        
    with open(f"data/Overcooked_tagged/{layout}/{id}_{score}.txt", "w") as f:
        for i in range(min(len(tagged_actions[0]), len(tagged_actions[1]))):
            f.write(state_graph[i] + "\n")
            joint_action = ast.literal_eval(trajectory[i]["action"])
            joint_action[0] = Action.ACTION_TO_CHAR[Action.ALL_ACTIONS[joint_action[0]]]
            joint_action[1] = Action.ACTION_TO_CHAR[Action.ALL_ACTIONS[joint_action[1]]]
            f.write("Alice: " + tagged_actions[0][i] + "\t" + joint_action[0] + "\n")
            f.write("Bob: " + tagged_actions[1][i] + "\t" + joint_action[1] + "\n")
    assert reward == 0
    return dataset

random.seed(42)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = args.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    files = glob.glob("data/Overcooked/*/*.csv")
    data = []
    
    for file in files:
        layout = file.split("/")[-2]
        reward = int(file.split("_")[-1].split(".")[0])
        id = int(file.split("/")[-1].split("_")[0])
        df = pandas.read_csv(file)
        df_dict_list = []
        for i in range(len(df)):
            df_dict_list.append(df.iloc[i].to_dict())
        data.extend(generate_dataset_dict_from_trajectory(df_dict_list, tokenizer, layout, reward, id))
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
    if "deepseek" in tokenizer.name_or_path.lower():
        train_df.to_parquet("data/Overcooked/DeepSeek/train.parquet")
        test_df.to_parquet("data/Overcooked/DeepSeek/test.parquet")
    else:
        train_df.to_parquet("data/Overcooked/Qwen/train.parquet")
        test_df.to_parquet("data/Overcooked/Qwen/test.parquet")