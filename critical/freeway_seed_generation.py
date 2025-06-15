from envs.minatar.environment import Environment
from envs.freeway import llm_state_builder, react_to_collision    

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

        for action in ['U', 'S', 'D']:
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

    return ['S'] * 101

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

        for action in ['U', 'S', 'D']:
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
def greedy(env: Environment, X): # greedy with X steps looking forward
    env.env.pos = 9
    step = 0
    while step < 100:
        state = llm_state_builder(env.env)
        action = react_to_collision(state, X)
        r, terminal = env.act(action)
        step += 1
        if r > 0:
            return step
        elif r < 0:
            return 101
    return step

import pandas as pd
import numpy as np
if __name__ == "__main__":
    env = Environment('freeway', sticky_action_prob=0)
    optimal_path = []
    logs = {
        'seed': [],
        'action': [],
        'greedy': []
    }
    for seed in range(1000, 3001):
        env.seed(seed)
        env.reset()
        best_action = bfs(env, max_steps=100)
        if best_action == []:
            continue
        optimal_path.append(best_action)
        logs['seed'].append(seed)
        logs['action'].append(best_action)
        X = 0
        for i in range(len(best_action)):
            env.seed(seed)
            env.reset()
            step = greedy(env, X)
            if step == len(best_action):
                break
            X += 1
        logs['greedy'].append(X)
        if X >= len(best_action):
            continue
        if X > 0:
            print(f"Seed: {seed}, Path: {best_action}, Greedy Level: {X}")
    df = pd.DataFrame(logs)
    df.to_csv('optimal_path.csv', index=False)
    df = pd.read_csv('optimal_path.csv')
    glvl = df['greedy'].tolist()
    
    import matplotlib.pyplot as plt
    counts, bins, _ = plt.hist(glvl, bins=max(glvl) - min(glvl), edgecolor='black')
    for i in range(len(bins) - 1):
        print(f"Range: {bins[i]} - {bins[i+1]}, X: {counts[i]}") 