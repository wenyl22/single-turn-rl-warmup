# This is Overcooked environment for one player cooperated with a script agent.
from envs.minatar.environments.overcooked_new.Overcooked_Env import Overcooked
from collections import defaultdict
from envs.minatar.environments.overcooked_new.src.overcooked_ai_py.mdp.actions import Action, Direction
shaped_info_keys = [
    "put_onion_on_X",
    "put_tomato_on_X",
    "put_dish_on_X",
    "put_soup_on_X",
    "pickup_onion_from_X",
    "pickup_onion_from_O",
    "pickup_tomato_from_X",
    "pickup_tomato_from_T",
    "pickup_dish_from_X",
    "pickup_dish_from_D",
    "pickup_soup_from_X",
    "USEFUL_DISH_PICKUP", # counted when #taken_dishes < #cooking_pots + #partially_full_pots and no dishes on the counter
    "SOUP_PICKUP", # counted when soup in the pot is picked up (not a soup placed on the table)
    "PLACEMENT_IN_POT", # counted when some ingredient is put into pot
    "viable_placement",
    "optimal_placement",
    "catastrophic_placement",
    "useless_placement",
    "potting_onion",
    "potting_tomato",
    "delivery",
]
import numpy as np
class Env:
    def __init__(self, ramping=None):
        self.all_args = None
        self.run_dir = None
        self.random = np.random.RandomState()
        self.seed = 42

    def reset(self):
        self.gym_env = Overcooked(self.all_args, self.run_dir, featurize_type=("bc", "bc"))
        eval_obs, _, _ = self.gym_env.reset(True)

        self.reward = 0
        self.game_turn = 0
        self.terminal = False

        # self.eval_env_infos = defaultdict(list)

    def act(self, a):
        self.game_turn += 1
        if a == "U":
            action = Direction.SOUTH
        elif a == "D":
            action = Direction.NORTH
        elif a == "L":
            action = Direction.WEST
        elif a == "R":
            action = Direction.EAST
        elif a == "I":
            action = Action.INTERACT
        else:
            action = Action.STAY
        self.gym_env.script_agent[0].next_action = action
        eval_ob, eval_share_ob, eval_reward, eval_done, eval_info, eval_available_action = self.gym_env.step([[0], [0]])
        #print(eval_reward, eval_done)
        self.reward += sum(eval_reward[0])
        self.terminal = eval_done[0]
        # for a in range(self.num_agents):
        #     for i, k in enumerate(shaped_info_keys):
        #         self.eval_env_infos[f'eval_ep_{k}_by_agent{a}'].append(eval_info['episode']['ep_category_r_by_agent'][a][i])
        #     self.eval_env_infos[f'eval_ep_sparse_r_by_agent{a}'].append(eval_info['episode']['ep_sparse_r_by_agent'][a])
        #     self.eval_env_infos[f'eval_ep_shaped_r_by_agent{a}'].append(eval_info['episode']['ep_shaped_r_by_agent'][a])
        # self.eval_env_infos['eval_ep_sparse_r'].append(eval_info['episode']['ep_sparse_r'])
        # self.eval_env_infos['eval_ep_shaped_r'].append(eval_info['episode']['ep_shaped_r'])
        return self.reward, self.terminal
    def state_string(self):
        ret = self.gym_env.base_mdp.state_string(self.gym_env.base_env.state)
        ret = ret.split('\n')
        ret = ret[::-1]
        ret = "\n".join(ret)
        ret = ret.replace("↑", "x").replace("↓", "y")
        ret = ret.replace("x", "↓").replace("y", "↑")
        return ret
    