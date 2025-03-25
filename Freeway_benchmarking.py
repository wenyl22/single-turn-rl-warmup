import os
import sys
import time 
import numpy as np 
import argparse
from minatar.environment import Environment
from Freeway_agent import LLMAgent, prompt_builder

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
parser.add_argument('--max_new_tokens', type=int, default=2000)
parser.add_argument('--token_per_tick', type=int, default=2000)
args = parser.parse_args()

model_name = args.model_name
max_new_tokens = args.max_new_tokens
token_per_tick = args.token_per_tick
print(f'Benchmarking model: {model_name}')
def game_loop(ModelName, MaxNewTokens, TokenPerTick, RunType="serial", seed=42):
    env = Environment('freeway', sticky_action_prob=0)
    env.reset()
    env.seed(seed)
    agent = LLMAgent(ModelName, MaxNewTokens, TokenPerTick, RunType, seed)
    start_time = time.time()
    terminal = False
    reward = 0
    game_turn = 0
    while (reward < 0.5) and (terminal is False):
        action = 0
        if env.env.move_timer == 0:
            state_for_llm = prompt_builder(env.env)
            action = agent.get_next_move(state_for_llm)
            if "stay" in action.lower():
                action = 0
            elif "up" in action.lower(): 
                # 1. move up for llm is (y + 1)
                # 2. Freeway are numbered as 0 to 9, LLM starts at y = 0; but the game controller numbers the freeway in a reverse order.
                # 3. "up"(2) in the game controller brings LLM to freeway k - 1
                # 4. So these actually match
                action = 2
            elif "down" in action.lower():
                action = 4
        reward, terminal = env.act(action)
        game_turn += 1
        if reward > 0.5:
            print(f"Get to the otherside in {game_turn} actions!")
            break
        if terminal or (game_turn > 150):
            print("Fail to get to the otherside in required turns")
            break
    game_time = time.time() - start_time
    print(f"Game took: {game_time} seconds")
    return seed, game_turn, game_time

if __name__ == '__main__':
    game_turns = []
    game_times = []
    NUM_TRIALS = 5
    for i in range(NUM_TRIALS):
        _, nt, game_time = game_loop(model_name, max_new_tokens, token_per_tick, "serial", 42 + i)
        game_turns.append(nt)
        game_times.append(game_time)
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    with open(f'logs/freeway/{model_name.split('/')[-1]}/summarize_{max_new_tokens}_{token_per_tick}_{time_stamp}.txt', 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Mean Turns: {np.mean(game_turns)}\n')
        f.write(f'Standard Error: {np.std(np.array(game_turns)) / np.sqrt(NUM_TRIALS)}\n')
        f.write(f'Turns: {game_turns}\n')
        f.write(f"---------------------------------------\n")
        f.write(f'Mean Game Time: {np.mean(game_times)}\n')
        f.write(f'Game Times: {game_times}\n')
    