# single step seed generation: 200
# 30% needs planning
#   - greedy level >= 1, only safe action is D or S. If safe, D and S should be both optimal, makes no difference.
# 70% doesn't need planning:
#   - greedy level = 0
#   - Safe action can be U, D, S. 
#   - For 35%, select data points where only one action is safe.
#   - For 35%, select data points where all actions are safe, but only one action is optimal.

# metric:
# 1. Safe : whether taking this single step will lead to a must collision
# 2. Optimal: whether this single step can be part of the optimal path, with partial observabiltiy
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from envs.minatar.environment import Environment
from freeway_seed_generation import bfs, greedy
from envs.freeway import llm_state_builder, react_to_collision, tick, po_navigate, check_collision
import pandas as pd
env = Environment('freeway', sticky_action_prob=0)
seeds = {
    'seed': [], #(seed, time_step, pos, correct_hint)
    'render': [],
    'need_planning': [], # whether the action needs planning
    'safe_actions': [],
    'optimal_actions': [],
}
cnt = [0, 0, 0]
for seed in range(1001, 3000):
    if sum(cnt) == 200:
        break
    env.seed(seed)
    env.reset()
    best_action = bfs(env, max_steps=100)
    if len(best_action) > 100:
        continue
    nenv = env.deep_copy()
    greedy_actions = greedy(nenv, 10)
    if greedy_actions > len(best_action):
        continue
    for i in range(len(best_action)):
        env.act(best_action[i])
        state_for_llm = llm_state_builder(env.env)
        greedy_action = react_to_collision(state_for_llm, 0)
        opt_action = react_to_collision(state_for_llm, 10)
        safe_actions = []
        optimal_actions = []
        po_greedy = po_navigate(state_for_llm)
        for a in ['U', 'D', 'S']:
            if check_collision(state_for_llm, 10, a):
                continue
            new_state_for_llm, r = tick(state_for_llm, a)
            safe_actions.append(a)
            a_po_navigate = po_navigate(new_state_for_llm)
            if len(a_po_navigate) == len(po_greedy) - 1:
                optimal_actions.append(a)
        if greedy_action != opt_action: # needs planning
            if cnt[0] >= 60:
                continue
            cnt[0] += 1
            seeds['seed'].append((seed, i + 1, state_for_llm['player_states'], po_greedy))
            seeds['render'].append("\n" + env.env.state_string() + "\n")
            seeds['need_planning'].append(True)
            seeds['safe_actions'].append(safe_actions)
            seeds['optimal_actions'].append(optimal_actions)
        elif len(safe_actions) == 1: # needs basic judgement for safe
            if cnt[1] >= 70:
                continue
            cnt[1] += 1
            seeds['seed'].append((seed, i + 1, state_for_llm['player_states'], po_greedy))
            seeds['render'].append("\n" + env.env.state_string() + "\n")
            seeds['need_planning'].append(False)
            seeds['safe_actions'].append(safe_actions)
            seeds['optimal_actions'].append(optimal_actions)
        elif len(optimal_actions) == 1: # needs basic judgement for optimal
            if cnt[2] >= 70:
                continue
            cnt[2] += 1
            seeds['seed'].append((seed, i + 1, state_for_llm['player_states'], po_greedy))
            seeds['render'].append("\n" + env.env.state_string() + "\n")
            seeds['need_planning'].append(False)
            seeds['safe_actions'].append(safe_actions)
            seeds['optimal_actions'].append(optimal_actions)
        else:
            continue
        break
    print(f"Seed {seed} done, cnt: {cnt}, total: {sum(cnt)}")

# dump seed to file
df = pd.DataFrame(seeds)
df.to_csv('critical/SingleStepDataset.csv', index=False)
            
            