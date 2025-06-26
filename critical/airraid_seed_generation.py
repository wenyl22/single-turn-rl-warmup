import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from envs.minatar.environment import Environment
from collections import deque
import pandas as pd

def bfs(env, max_steps = None):
    if max_steps is None:
        max_steps = env.env.MAX_TURN
    queue = deque()
    queue.append((env.env.pos, 0))
    visited = {}
    env_list = [env]

    for i in range(1, max_steps):
        env_copy = env_list[i - 1].deep_copy()
        env_copy.act('S')
        env_list.append(env_copy)

    while queue:
        pos, steps = queue.popleft()
        rewards = visited.get((pos, steps), 0)
        if steps >= max_steps:
            continue
        for action in ['L', 'S', 'R']:
            new_env = env_list[steps].deep_copy()
            new_env.env.pos = pos
            r, terminal = new_env.env.act(action)
            if (new_env.env.pos, steps + 1) not in visited:
                visited[(new_env.env.pos, steps + 1)] = 0
                queue.append((new_env.env.pos, steps + 1))
            visited[(new_env.env.pos, steps + 1)] = max(visited[(new_env.env.pos, steps + 1)], rewards + r)

    max_reward = -1
    for (pos, steps), reward in visited.items():
        if steps == max_steps and reward > max_reward:
            max_reward = reward
    return max_reward

def dp_with_horizon(env, horizon):
    tars = []
    for j in range(10):
        ships = env.env.space_ships[j]
        if isinstance(ships, tuple):
            ships = [ships]
        for (x, y, speed, reward) in ships:
            if y <= 0 or (y + speed - 1) // speed > horizon:
                continue
            tars.append((x, y, (y + speed - 1) // speed, reward))
    tars.append((env.env.pos, 0, 0, 0))
    tars = sorted(tars, key=lambda x: x[2])
    dp = [-1 for _ in range(len(tars))]
    dp[0] = 0
    max_r = 0
    for i in range(len(tars)):
        for j in range(0, i):
            if tars[i][2] - tars[j][2] >= abs(tars[i][0] - tars[j][0]):
                dp[i] = max(dp[i], dp[j] + tars[i][3])
        if dp[i] > dp[max_r]:
            max_r = i
    return dp[max_r]



def greedy(env, max_steps = None, horizon = None):
    if max_steps is None:
        max_steps = env.env.MAX_TURN
    if horizon is None:
        horizon = env.env.MAX_TURN
    actions = []
    for step in range(max_steps):       
        max_reward = -1
        best_action = None
        for action in ['L', 'R', 'S']:
            cur_env = env.deep_copy()
            r, t = cur_env.act(action)
            next_reward = dp_with_horizon(cur_env, horizon)
            if next_reward + r > max_reward:
                max_reward = next_reward + r
                best_action = action
            elif next_reward + r == max_reward:
                return None, None
        actions.append(best_action)
        env.act(best_action)
    return env.env.reward, actions    

if __name__ == "__main__":
    env = Environment('airraid', sticky_action_prob=0)
    optimal_path = []
    logs = {
        'seed': [],
        'action': [],
    }
    cnt = 0
    seed_list = {}
    for seed in range(1, 600):
        env.seed(seed)
        env.reset()
        reward, actions = greedy(env)
        if actions is not None:
            env.seed(seed)
            env.reset()
            max_reward = bfs(env)
            print(f'Seed: {seed}, Actions: {" ".join(actions)}, Reward: {reward}')
            print(reward, max_reward)
            if reward != max_reward:
                continue
            optimal_path.append((seed, actions, reward))
            logs['seed'].append(seed)
            logs['action'].append(' '.join(actions))
            env.seed(seed)
            env.reset()
            _reward, _actions = greedy(env, horizon = 5)
            if _actions is not None:
                print(f'Seed: {seed}, Actions (5-step): {" ".join(_actions)}, Reward: {_reward}')
            else:
                print(f'Seed: {seed}, No valid actions found in 5-step horizon.')
            seed_list[cnt] = seed
            cnt += 1
    print(seed_list)
    df = pd.DataFrame(logs)
    df.to_csv('optimal_path.csv', index=False)