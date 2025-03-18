import glob
import random
import argparse
import pandas
from typing import List, Dict, Any
from transformers import AutoTokenizer
from Freeway_prompts import MESSAGE
import ast
from minatar.environments.freeway import Env
import os


def prompt_builder(env: dict):
    player_states = (env["pos"], "I can move in this turn" if env["move_timer"] == 0 else f"I cannot move in the following {env["move_timer"]} turns")
    car_states = []
    for car in env["cars"]:
        dir = "towards"
        if car[0] > 4 and car[3] > 0:
            dir = "away from"
        elif car[0] < 4 and car[3] < 0:
            dir = "away from"
        car_states.append(
            (car[1], car[0] - 4, dir, car[2])
        )
    available_actions = []

    if env["move_timer"] == 0:
        if env["pos"] > 0:
            available_actions.append("Move down (to Freeway " + str(env["pos"] - 1) + ")")
        if env["pos"] < 9:
            available_actions.append("Move up (to Freeway " + str(env["pos"] + 1) + ")")
    available_actions.append("Stay in the same freeway")
    state_for_llm = {
        'player_states': player_states,
        'car_states': car_states,
        'available_actions': available_actions
    }
    return state_for_llm

def get_available_actions(state_for_llm):
    available_actions_list = []
    description = ''
    for i, action in enumerate(state_for_llm['available_actions']):
        available_actions_list.append(f'{chr(65+i)} {action}')
    for action in available_actions_list:
        description += f'{action}\n'
    return description

def state_to_description(state_for_llm):
    description = f"I am on Freeway {state_for_llm["player_states"][0]}. {state_for_llm["player_states"][1]}."
    for car in state_for_llm['car_states']:
        description += f"There is a car at $x = {car[1]}$ on Freeway {car[0]}. It's moving {car[2]} me and will move 1 unit forward in {car[3]} turns.\n"
    
    description += f'Available actions:\n{get_available_actions(state_for_llm)}'
    return description

def generate_dataset_dict_from_trajectory(trajectory: List[Dict[str, Any]], tokenizer: AutoTokenizer, score: int, file: str):
    print(file)
    dataset = []
    env = Env()
    for i, step in enumerate(trajectory):
        # step["state"] is a dictionary stored in string format
        # turn it back into a dictionary
        state_dict = ast.literal_eval(step["game_state"])
        env.from_dict(state_dict)
        state_for_llm = prompt_builder(state_dict)
        description = state_to_description(state_for_llm)
        action = step["action"]
        if action != "u" and action != "d":
            action = "n"
        env.from_dict(state_dict)
        r, d = env.act(action)
        score -= r
        # print(state_dict)
        # print(action)
        # print(r, step["reward"])
        # print(file)
        # if i < len(trajectory) - 1:
        #     print(trajectory[i+1]["game_state"])
        assert r == int(step["reward"])
        valid = False
        for valid_action in state_for_llm['available_actions']:
            if action == "u" and "Move down" in valid_action:
                valid = True
                action = valid_action
                break
            if action == "d" and "Move up" in valid_action:
                valid = True
                action = valid_action
                break
        if not valid:
            action = "Stay in the same freeway"
        prompt = MESSAGE + [{"role": "user", "content": description}]
        if "deepseek" in tokenizer.name_or_path.lower():
            prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        else:
            prompt = tokenizer.apply_chat_template(prompt, add_special_tokens=False, tokenize=False)
            prompt += '<|im_start|>assistant\n<think>'
        # print("Prompt:", prompt)
        # print("Action:", action)
        # print("Available actions:", state_for_llm['available_actions'])
        # exit(0)
        dataset.append(
            {
                "prompt": prompt,
                "solution": action,
                "available_actions": state_for_llm['available_actions'],
            }
        )
    assert score == 0
    return dataset

random.seed(42)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    args = args.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    files = glob.glob("data/Freeway/speed_1/*.csv")
    data = []
    for file in files:
        df = pandas.read_csv(file)
        score = int(file.split("_")[-1].split(".")[0])
        df_dict_list = []
        for i in range(len(df)):
            df_dict_list.append(df.iloc[i].to_dict())
        data.extend(generate_dataset_dict_from_trajectory(df_dict_list, tokenizer, score, file))
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
    if not os.path.exists("data/Freeway"):
        os.makedirs("data/Freeway")    
    if "deepseek" in tokenizer.name_or_path.lower():
        if not os.path.exists("data/Freeway/DeepSeek"):
            os.makedirs("data/Freeway/DeepSeek")
        train_df.to_parquet("data/Freeway/DeepSeek/train.parquet")
        test_df.to_parquet("data/Freeway/DeepSeek/test.parquet")
    else:
        if not os.path.exists("data/Freeway/Qwen"):
            os.makedirs("data/Freeway/Qwen")
        train_df.to_parquet("data/Freeway/Qwen/train.parquet")
        test_df.to_parquet("data/Freeway/Qwen/test.parquet")
