from envs.minatar.environment import Environment
from envs.visualize_utils import generate_gif_from_string_map

# Bfs to find the best action
def bfs(env, max_steps=100):
    from collections import deque

    queue = deque()
    queue.append((env.env.pos, 0, []))
    visited = set()
    env_list = [env]
    for i in range(1, max_steps):
        env_copy = env_list[i - 1].deep_copy()
        env_copy.act(0)
        env_list.append(env_copy)

    while queue:
        pos, steps, actions = queue.popleft()
        if steps >= max_steps:
            continue

        for action in [2, 0, 4]:
            new_env = env_list[steps].deep_copy()
            new_env.env.pos = pos
            r, terminal = new_env.act(action)

            if r > 0:
                return actions + [action]
            if r < 0:
                continue
            if (new_env.env.pos, steps + 1) in visited:
                continue
            visited.add((new_env.env.pos, steps + 1))
            queue.append((new_env.env.pos, steps + 1, actions + [action]))

    return []
def all_paths(env, optimal_step):
    from collections import deque
    queue = deque()
    queue.append((env.env.pos, 0, [], 0))
    visited = set()
    env_list = [env]
    sols = []
    for i in range(1, optimal_step):
        env_copy = env_list[i - 1].deep_copy()
        env_copy.act(0)
        env_list.append(env_copy)

    while queue:
        pos, steps, actions, tm = queue.popleft()
        if pos == 0:
            print(f"Found a solution: {actions}, tm: {tm}")
            sols.append((actions, tm))
            continue
        if steps >= optimal_step:
            continue

        for action in [2, 0, 4]:
            new_env = env_list[steps].deep_copy()
            new_env.env.pos = pos
            r, terminal = new_env.act(action)
            if r > 0:
                new_env.env.pos = 0
            if r < 0:
                new_tm = tm+1
            else:
                new_tm = tm
            if (new_env.env.pos, steps + 1, new_tm) in visited:
                continue
            visited.add((new_env.env.pos, steps + 1, new_tm))
            queue.append((new_env.env.pos, steps + 1, actions + [action], new_tm))

    return sols
from envs.freeway import supervise_collision, llm_state_builder, react_to_collision
def greedy(env: Environment):
    env.env.pos = 9
    step = 0
    while step < 100:
        state = llm_state_builder(env.env)
        action = react_to_collision(state)
        action = 2 if action == 1 else 4 if action == 2 else 0
        r, terminal = env.act(action)
        step += 1
        if r > 0:
            return step
    return step

        
        
import pandas as pd
import numpy as np
if __name__ == "__main__":
    env = Environment('freeway', sticky_action_prob=0)
    env.env.difficulty = 8
    env.env.pos = 0
    env.env.special_pattern = True
    optimal_path = []
    logs = {
        'seed': [],
        'action': [],
        'greedy': []
    }
    # for seed in range(1001, 3000):
    #     env.seed(seed)
    #     env.env.special_pattern = True
    #     env.reset()
    #     best_action = bfs(env, max_steps=100)
    #     if best_action == []:
    #         continue
    #     optimal_path.append(best_action)
    #     logs['seed'].append(seed)
    #     logs['action'].append(best_action)
    #     env.seed(seed)
    #     env.env.special_pattern = True
    #     env.reset()
    #     step = greedy(env)
    #     logs['greedy'].append(step)
    #     if step >= 100 > 25 > len(best_action):
    #         print(f"Seed: {seed}, Path: {best_action}, Greedy: {step} > {len(best_action)}")
    # df = pd.DataFrame(logs)
    # df.to_csv('optimal_path.csv', index=False)
    df = pd.read_csv('optimal_path.csv')
    # optimal_paths = []
    # for i in range(len(df)):
    #     temp = eval(df['action'][i])
    #     while temp[0] == 0:
    #         temp = temp[1:]
    #     optimal_paths.append(temp)
    #     if len(temp) == 22:
    #         print(f"Seed: {df['seed'][i]}, Path: {temp}")
    # length = []
    # for i in range(len(optimal_paths)):
    #     length.append(len(optimal_paths[i]))
    # import matplotlib.pyplot as plt
    # counts, bins, _ = plt.hist(length, bins=max(length) - min(length), edgecolor='black')
    # for i in range(len(bins) - 1):
    #     print(f"Range: {bins[i]} - {bins[i+1]}, Height: {counts[i]}") 
    # plt.savefig('histogram.png')
    # font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    seeds = [1069, 1093, 1447, 1953, 1536, 1798, 1858, 2408]

    seed_mapping_list = {}

    for i, seed in enumerate(seeds):
        env.seed(seed)
        env.reset()
        best_action = bfs(env, max_steps=100)
        print(f"--------------------Seed{seed}--------------------")
        env.seed(seed)
        env.reset()
        sols = all_paths(env, len(best_action))
        env.seed(seed)
        env.reset()
        string_maps = []
        flag = False
        tmp = 0
        
        for action in best_action:
            str_mp = env.env.state_string()
            string_maps.append(str_mp)
            env.act(action)
            if seed == 1798:
                if tmp < 5:
                    string_maps = []
                    tmp += 1
                    continue
            # print(str_mp)
            flag = True
        seed_mapping_list[i] = (seed, len(string_maps), tmp)
        # generate_gif_from_string_map(string_maps, f"example_gifs/{seed}.gif", font_path=font_path, font_size=20)
    print(seed_mapping_list)

