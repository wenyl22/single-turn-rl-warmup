import glob
import random
import argparse
import pandas
from typing import List, Dict, Any
from transformers import AutoTokenizer
from envs.prompts.freeway import LLM_SYSTEM_PROMPT, LLM_BASE_PROMPT
import ast
from envs.minatar.environments.freeway import Env
import os


def prompt_builder(env: dict):
    player_states = (9 - env["pos"], "You can move in this turn" if env["move_timer"] == 0 else f"I cannot move in the following {env["move_timer"]} turns")
    car_states = []
    for car in env["cars"]:
        # dir = "towards"
        # if car[0] > 4 and car[3] > 0:
        #     dir = "away from"
        # elif car[0] < 4 and car[3] < 0:
        #     dir = "away from"
        dir = 'left' if car[3] < 0 else 'right'
        speed = 12 // abs(car[3])
        pos = 12 * (car[0] - 4)
        if dir == 'left':
            pos -= (abs(car[3]) - car[2] - 1) * speed
        else:
            pos += (abs(car[3]) - car[2] - 1) * speed
        assert car[2] < abs(car[3])
        car_states.append(
            (9 - car[1], pos, dir, speed)
        )
    car_states = car_states[::-1]
    available_actions = []

    if env["move_timer"] == 0:
        assert env["pos"] > 0
        available_actions.append("Move up to Freeway " + str(9 - env["pos"] + 1))
        if env["pos"] < 9:
            available_actions.append("Move down to Freeway " + str(9 - env["pos"] - 1))
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
    description = f"-**Your position**: (0, {state_for_llm["player_states"][0]}).\n"
    description += '-**Cars on each freeway**:\n'
    for car in state_for_llm['car_states']:
        span = 11 if car[2] == 'left' else -11
        description += f"\t-**Freeway {car[0]}**: head at **x = {car[1]}**, tail at **x = {car[1] + span}**, direction = {car[2]}, speed = {car[3]}.\n"
    description += f'Available actions:\n{get_available_actions(state_for_llm)}'
    return description

def generate_dataset_dict_from_trajectory(trajectory: List[Dict[str, Any]], tokenizer: AutoTokenizer, score: int, file: str):
    print(file)
    dataset = []
    env = Env()
    for i, step in enumerate(trajectory):
        if i == 0:
            continue
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
        if i < len(trajectory) - 1:
            new_state_dict = ast.literal_eval(trajectory[i+1]["game_state"])
            assert env.pos == new_state_dict["pos"]
        assert r == int(step["reward"])
        valid = False
        for valid_action in state_for_llm['available_actions']:
            if action == "u" and "Move up" in valid_action:
                valid = True
                action = valid_action
                break
            if action == "d" and "Move down" in valid_action:
                valid = True
                action = valid_action
                break
        if not valid:
            action = "Stay in the same freeway"
        prompt = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": LLM_BASE_PROMPT + description},
        ]
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
    args.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    args = args.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    files = sorted(glob.glob("data/Freeway/speed_1/*.csv"))
    data = []
    for i, file in enumerate(files):
        df = pandas.read_csv(file)
        score = int(file.split("_")[-1].split(".")[0])
        df_dict_list = []
        for i in range(len(df)):
            df_dict_list.append(df.iloc[i].to_dict())
        data.extend(generate_dataset_dict_from_trajectory(df_dict_list, tokenizer, score, file))
    #     break
    # # data to csv
    # df = pandas.DataFrame(data)
    # df.to_csv("data/Freeway/Freeway.csv", index=False)
    print(len(data))
    # print "up", "down", "stay" in data[:]["solution"]
    print("up:", sum([1 if "up" in d["solution"].lower() else 0 for d in data]))
    print("down:", sum([1 if "down" in d["solution"].lower() else 0 for d in data]))
    print("stay:", sum([1 if "stay" in d["solution"].lower() else 0 for d in data]))
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    split = int(0.5 * len(data))
    train_data = data[:split]
    test_data = data[split:]
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
