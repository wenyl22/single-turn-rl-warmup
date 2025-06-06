from envs.minatar.environment import Environment
from envs.airraid import llm_state_builder, tick
import pandas as pd
from air_test_accuracy import greedy as accuracy_greedy

def greedy(state_for_llm):
    ships = state_for_llm["reward_states"]
    pos = state_for_llm["player_states"]
    tars = []
    for (x, y, speed, reward) in ships:
        if y <= 0:
            continue
        tars.append((x, y, (y + speed - 1) // speed, reward))
    tars.append((pos, 0, 0, 0))
    tars = sorted(tars, key=lambda x: x[2])
    dp = [0] * 32
    cnt = [0] * 32
    cnt[0] = 1
    pre = [0] * 32
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
    if max_cnt > 1 or max_r == 0:
        return None, None, None
    collect_list = []
    max_reward = dp[max_r]
    while True:
        collect_list.append((tars[max_r][0], tars[max_r][2], tars[max_r][3]))
        if max_r == 0:
            break
        max_r = pre[max_r]
    if len(collect_list) <= 3:
        return None, None, None
    collect_list.reverse()
    actions = []
    for i in range(len(collect_list) - 1):
        cur, t_cur, _ = collect_list[i]
        nxt, t_nxt, _ = collect_list[i + 1]
        if cur < nxt:
            actions.append('R' * (nxt - cur) + 'S' * (t_nxt - t_cur - nxt + cur))
        elif cur > nxt:
            actions.append('L' * (cur - nxt) + 'S' * (t_nxt - t_cur - cur + nxt))
        else:
            actions.append('S' * (t_nxt - t_cur))
    return actions, max_reward, len(collect_list)

def generate_dataset():   
    """
    1. randomly generate a seed and let the time flow for some random steps
    2. Choose a state, such that the maximum reward is unique, and the 
    """
    logs = {
        "seed": [],
        "actions": [],
        "reward": [],
        "num": [],
        "render": [],
    }
    env = Environment('airraid', sticky_action_prob=0)
    for seed in range(1000):
        env.seed(seed)
        env.reset()
        for _ in range(100):
            env.act(0)
        env.env.pos = (seed * seed) % 9
        mp = env.env.state_string()
        actions, reward, collected_num = greedy(env)
        actions = "".join(actions)
        if actions is None or len(actions) <= 5:
            continue
        logs["seed"].append(seed)
        logs["actions"].append(actions)
        logs["reward"].append(reward)
        logs["num"].append(collected_num - 1)
        logs["render"].append("\n" + mp + "\n")
        print("collected seed:", len(logs["seed"]))
        if len(logs["seed"]) >= 100:
            break
    df = pd.DataFrame(logs)
    df.to_csv("data/airraid/dataset.csv", index=False)

def generate_new_dataset():
    logs = {
        "seed": [],
        "actions": [],
        "reward": [],
        "num": [],
        "mapping": [],
        "render": [],
    }
    for seed in range(3000):
        env = Environment('airraid', sticky_action_prob=0)
        env.seed(seed)
        env.reset()
        for _ in range(80):
            env.act(0)
        env.env.pos = (seed * seed) % 9
        mp = env.env.state_string()
        state_for_llm = llm_state_builder(env.env)
        actions, max_reward, collected_num = greedy(state_for_llm)
        if actions is None or len(actions) <= 5:
            continue
        actions = "".join(actions)
        mapping = {}
        flag = True
        for first_action in ["L", "R", "S"]:
            n_state_for_llm, n_r = tick(state_for_llm, first_action)
            _, n_reward, _ = accuracy_greedy(n_state_for_llm)
            if n_reward >= max_reward and first_action != actions[0]:
                flag = False
                break
            mapping[first_action] = n_r + n_reward
        if not flag:
            continue
        logs["seed"].append(seed)
        logs["actions"].append(actions)
        logs["reward"].append(max_reward)
        logs["num"].append(collected_num - 1)
        logs["mapping"].append(mapping)
        logs["render"].append("\n" + mp + "\n")
        print("collected seed:", len(logs["seed"]))
        if len(logs["seed"]) >= 100:
            break
    df = pd.DataFrame(logs)
    df.to_csv("data/airraid/new_dataset.csv", index=False)

if __name__ == "__main__":
    # generate_dataset()
    generate_new_dataset()
