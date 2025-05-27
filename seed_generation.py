from envs.minatar.environment import Environment
from collections import deque
from envs.utils.visualize_utils import generate_gif_from_string_map
import pandas as pd
mapping = {
    0: 'S',
    1: 'L',
    3: 'R',
}

def bfs(env, max_steps=50):
    queue = deque()
    queue.append((env.env.pos, 0))
    visited = {}
    env_list = [env]
    for i in range(1, max_steps):
        env_copy = env_list[i - 1].deep_copy()
        env_copy.act(0)
        env_list.append(env_copy)

    while queue:
        pos, steps = queue.popleft()

        rewards = visited.get((pos, steps), 0)
        if steps >= max_steps:
            continue

        for action in [0, 1, 3]:
            new_env = env_list[steps].deep_copy()
            new_env.env.pos = pos
            r, terminal = new_env.act(action)
            if (new_env.env.pos, steps + 1) not in visited:
                visited[(new_env.env.pos, steps + 1)] = 0
                queue.append((new_env.env.pos, steps + 1))
            visited[(new_env.env.pos, steps + 1)] = max(visited[(new_env.env.pos, steps + 1)], rewards + r)

    max_reward = -1
    for (pos, steps), reward in visited.items():
        if steps == max_steps and reward > max_reward:
            max_reward = reward
    return max_reward

def greedy(env, max_steps=50):
    """
    Greedily find the best action in the environment.
    """
    actions = []
    rewards = 0
    string_maps = [env.env.state_string()]
    for _ in range(max_steps):
        ships = env.env.space_ships
        tars = []
        for (x, y, speed, reward) in ships:
            if y <= 0:
                continue
            # print(x, y, speed, reward)
            tars.append((x, y, (y + speed - 1) // speed, reward))
            
        tars.append((env.env.pos, 0, 0, 0))  # Add player position with no reward
        tars = sorted(tars, key=lambda x: x[2])
        dp = [0] * 12
        cnt = [0] * 12
        cnt[0] = 1
        pre = [0] * 12
        max_r = 0
        max_cnt = 0
        for i in range(len(tars)):
            for j in range(0, i):
                if cnt[j] == 0:
                    continue
                if tars[i][2] - tars[j][2] >= abs(tars[i][0] - tars[j][0]):
                    if dp[i] < dp[j] + tars[i][3]:
                        cnt[i] = cnt[j]
                        pre[i] = j
                    elif dp[i] == dp[j] + tars[i][3]:
                        cnt[i] += cnt[j]
                    dp[i] = max(dp[i], dp[j] + tars[i][3])
            if dp[i] > dp[max_r]:
                max_r = i
                max_cnt = cnt[i]
            elif dp[i] == dp[max_r]:
                max_cnt += cnt[i]
        # print (env.env.pos, tars, dp, max_cnt, dp[max_r])
        if max_cnt > 1 or max_r == 0:
            return None, None, None
        while pre[max_r] != 0:
            max_r = pre[max_r]
        if tars[max_r][0] == env.env.pos:
            action = 0
        elif tars[max_r][0] < env.env.pos:
            action = 1
        else:
            action = 3
        actions.append(mapping[action])
        r, t = env.act(action)
        rewards += r
        string_maps.append(env.env.state_string())
        # print (env.env.state_string()+'\n', mapping[action], r, rewards)
    return actions, rewards, string_maps
if __name__ == "__main__":
    env = Environment('airraid', sticky_action_prob=0)
    optimal_path = []
    logs = {
        'seed': [],
        'action': [],
    }
    cnt = 0
    seed_list = {}
    for seed in range(1, 6000):
        env.seed(seed)
        env.reset()
        actions, reward, string_maps = greedy(env)
        if actions is not None:
            env.seed(seed)
            env.reset()
            max_reward = bfs(env)
            if reward != max_reward:
                continue

            optimal_path.append((seed, actions, reward))
            logs['seed'].append(seed)
            logs['action'].append(' '.join(actions))
            print(f'Seed: {seed}, Actions: {" ".join(actions)}, Reward: {reward}')
            seed_list[cnt] = seed
            cnt += 1
            # generate_gif_from_string_map(string_maps, f'optimal_path_{seed}.gif')
        # else:
        #     print(f'Seed: {seed}, Not a valid seed.')
    print(seed_list)
    df = pd.DataFrame(logs)
    df.to_csv('optimal_path.csv', index=False)
