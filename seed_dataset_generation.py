from envs.minatar.environment import Environment
from envs.freeway import llm_state_builder, react_to_collision
import pandas as pd
import random

def tick(state: dict):
    for i, car in enumerate(state['car_states']):
        ncar = [car[0], car[1], car[2], car[3], car[4]]
        if car[2] == 'left':
            ncar[1] -= ncar[3]
        else:
            ncar[1] += ncar[3]
        state['car_states'][i] = ncar
    return state

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

def greedy(env: Environment, X): # greedy with X steps looking forward
    step = 0
    act_list = []
    while step < 100:
        state = llm_state_builder(env.env)
        action = react_to_collision(state, X)
        action = 2 if action == 1 else 4 if action == 2 else 0
        r, terminal = env.act(action)
        act_list.append(action)
        step += 1
        if r > 0:
            return act_list
        elif r < 0:
            return [0] * 101
        state = tick(state)
    return act_list

def generate_dataset():        
    env = Environment('freeway', sticky_action_prob=0)
    env.env.difficulty = 8
    env.env.pos = 0
    env.env.special_pattern = True
    optimal_path = []
    logs = {
        'seed': [], # tuple (seed_value, start_pos, pass_time)
        'action': [], # an example of optimal path
        'greedy': [], # need to look forward how many steps
        'emergency': [], # where the first not 'U' action lies
    }
    for seed in range(1000, 3001):
        print(f"Seed: {seed}")
        env.seed(seed)
        env.reset()
        best_action = bfs(env, max_steps=100)
        if len(best_action) <= 9:
            continue
        env.seed(seed)
        env.reset()
        for pass_time in range(len(best_action) - 3):
            env.seed(seed)
            env.reset()
            for _ in range(pass_time):
                env.act(best_action[_])
            start_pos = env.env.pos
            state_string = env.env.state_string()
            greedy_steps = greedy(env, 20)
            if len(greedy_steps) == 101 or len(greedy_steps) == start_pos:
                continue
            GREEDY = 0
            for Xt in range(20):
                env.seed(seed)
                env.reset()
                for _ in range(pass_time):
                    env.act(best_action[_])
                step = greedy(env, Xt)
                GREEDY = Xt
                if len(step) == len(greedy_steps):
                    break
            # EMERGENCY is the first none-'U' action in greedy_steps
            EMERGENCY = min([i for i in range(len(greedy_steps)) if greedy_steps[i] != 2])
            if GREEDY > 0:
                # print(f"Seed: {seed}, Start Pos: {start_pos}, Pass Time: {pass_time}, Greedy: {GREEDY}, Emergency: {EMERGENCY}, length: {len(greedy_steps)}")
                # print(state_string)
                logs['seed'].append((seed, start_pos, pass_time))
                logs['action'].append(greedy_steps)
                logs['greedy'].append(GREEDY)
                logs['emergency'].append(EMERGENCY)
    df = pd.DataFrame(logs)
    df.to_csv("data/freeway/unfiltered_dataset.csv", index=False)
    # plot a table, showing the (Greedy, Emergency) pairs
import ast
def filter_dataset():
    df = pd.read_csv("data/freeway/unfiltered_dataset.csv")
    logs = {
        'seed': [ast.literal_eval(tp) for tp in df['seed'].tolist()],
        'action': [ast.literal_eval(tp) for tp in df['action'].tolist()],
        'greedy': df['greedy'].tolist(),
        'emergency': df['emergency'].tolist()
    }
    GE_pairs = {}
    Len = {}
    env = Environment('freeway', sticky_action_prob=0)
    for i in range(len(logs['greedy'])):
        g, e, l = logs['greedy'][i], logs['emergency'][i], len(logs['action'][i])
        if l > 16:
            continue
        if e > 2:
            continue
        seed = logs['seed'][i]
        env.seed(seed[0])
        env.reset()
        for _ in range(seed[2]):
            env.act(0)
        env.env.pos = seed[1]
        if not (g, e) in GE_pairs:
            GE_pairs[(g, e)] = []
        if not l in Len:
            Len[l] = []
        GE_pairs[(g, e)].append((seed[0], seed[1], seed[2], g, e, logs['action'][i]))
        Len[l].append((seed[0], seed[1], seed[2], g, e, logs['action'][i]))
    print("Greedy, Emergency pairs:")
    for k, v in GE_pairs.items():
        print(f"{k}: {len(v)}")
    print("Length pairs:")
    for k, v in sorted(Len.items(), key=lambda x: x[0]):
        print(f"{k}: {len(v)}")
    # Choose based on (G = 3, 2, 1, E = 0)
    # optimal path length as short as possible
    # do not choose situation repeated seed
    # Each G chooses at most 20 seeds
    chosen_seeds = set()
    chosen_seeds_list = {
        'seed': [],
        'render': []
    }
    random.seed(114514)
    for g in [3, 2, 1]:
        e = 0
        cnt = 0
        if (g, e) not in GE_pairs:
            continue
        seeds = GE_pairs[(g, e)]
        random.shuffle(seeds)
        for seed in seeds:
            if seed[0] in chosen_seeds:
                continue
            if g < 3 and len(seed[5]) < 12 or len(seed[5]) > 16:
                continue
            env.seed(seed[0])
            env.reset()
            for _ in range(seed[2]):
                env.act(0)
            env.env.pos = seed[1]
            chosen_seeds.add(seed[0])
            chosen_seeds_list['seed'].append(seed)
            chosen_seeds_list['render'].append('\n' + env.env.state_string() + '\n')
            cnt += 1
            if cnt >= 20:
                break
    for l in range(6, 16):
        seeds = Len[l]
        cnt = 0
        random.shuffle(seeds)
        for seed in seeds:
            if seed[0] in chosen_seeds:
                continue
            env.seed(seed[0])
            env.reset()
            for _ in range(seed[2]):
                env.act(0)
            env.env.pos = seed[1]
            chosen_seeds.add(seed[0])
            chosen_seeds_list['seed'].append(seed)
            chosen_seeds_list['render'].append('\n' + env.env.state_string() + '\n')
            cnt += 1
            if cnt >= 5:
                break

    df = pd.DataFrame(chosen_seeds_list)
    df.to_csv("data/freeway/dataset.csv", index=False)
        

if __name__ == "__main__":
    # generate_dataset()
    filter_dataset()