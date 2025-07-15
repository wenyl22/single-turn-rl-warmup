#!/usr/bin/env python
import random
import math
import hashlib
import logging
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
random.seed(42)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
print(sys.path)

from envs.minatar.environments.snake import Env
"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf

The State is a game where you have NUM_TURNS and at turn i you can make
a choice from an integeter [-2,2,3,-3]*(NUM_TURNS+1-i).  So for example in a game of 4 turns, on turn for turn 1 you can can choose from [-8,8,12,-12], and on turn 2 you can choose from [-6,6,9,-9].  At each turn the choosen number is accumulated into a aggregation value.  The goal of the game is for the accumulated value to be as close to 0 as possible.

The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  

In particular there are two models of best child that one can use 
"""

#MCTS scalar. Smaller scalar will increase exploitation, Larger will increase exploration. 
SCALAR = math.sqrt(2.0)


class State():
    def __init__(self, moves=[], initial_seed = 0):
        self.value = 0
        self.moves = moves
        self.terminal = False
        self.env = Env()
        self.initial_seed = initial_seed
        self.env.random = random.Random(self.initial_seed)
        self.env.seed = self.initial_seed
        self.env.reset()
        for m in self.moves:
            r, t = self.env.act(m)
            self.value += r
            self.terminal |= t
        self.MOVES = self.env.get_possible_actions()
        self.num_moves = len(self.MOVES)

    def next_state(self, move = None):
        if move is None:
            move = random.choice(self.MOVES)
        nextmove = move 
        next = State(self.moves + [nextmove], self.initial_seed)
        return next

    def reward(self):
        return self.value

    def __hash__(self):
        return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s="Value: %d; Moves: %s"%(self.value, self.moves)
        return s
    

class Node():
    def __init__(self, state, parent=None):
        self.visits = 0
        self.reward = 0.0    
        self.state = state
        self.children = []
        self.parent = parent    
    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f; Last move: %s"%(len(self.children), self.visits, self.reward, self.state.moves[-1] if len(self.state.moves) > 0 else "O")
        return s

def UCTSEARCH(budget, sim_level, plan_moves, root, num_moves_lambda = None):
    all_leaf = gather_leaf_children(root, plan_moves)
    for iter in range(int(budget)):
        front = TREEPOLICY(root, plan_moves, num_moves_lambda)
        reward = DEFAULTPOLICY(front.state, sim_level)
        BACKUP(front, reward)
    return BESTCHILD(root, 0, all_leaf)

def TREEPOLICY(node, plan_moves, num_moves_lambda):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal == False:
        level = len(node.state.moves)
        all_leaf = gather_leaf_children(node, plan_moves)
        new_moves = [_.state.moves[level:] for _ in all_leaf if _.visits == 0]
        if sum([c.visits for c in all_leaf]) == 0:
            return EXPAND(node, new_moves)
        elif random.uniform(0, 1) < .3:
            node = BESTCHILD(node, SCALAR, all_leaf)
        else:
            if new_moves != []:
                return EXPAND(node, new_moves)
            else:
                node = BESTCHILD(node, SCALAR, all_leaf)
    return node

def gather_leaf_children(node, plan_moves):
    if plan_moves == 0 or node.state.terminal:
        return [node]
    ret = []
    tried_moves = [c.state.moves[-1] for c in node.children]
    new_moves = [m for m in node.state.MOVES if m not in tried_moves]
    for m in new_moves:
        node.add_child(node.state.next_state(m))
    for c in node.children:
        children = gather_leaf_children(c, plan_moves - 1)
        ret.extend(children)
    return ret

def EXPAND(node, new_moves):
    if len(new_moves) == 0:
        raise Exception("No new moves to expand from node: %s"%node)
    moves = random.choice(new_moves)
    cur_node = node
    for m in moves:
        for c in cur_node.children:
            if c.state.moves[-1] == m:
                cur_node = c
                break
    return cur_node

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node, scalar, all_leaf):
    bestscore = -100
    bestchildren = []
    level = len(node.state.moves)
    for c in all_leaf:
        last_move = c.state.moves[level:]
        reward = c.reward
        visits = c.visits
        if c.visits == 0:
            continue
        exploit = reward / visits
        explore = math.sqrt(2.0 * math.log(node.visits) / float(visits))
        score = exploit + scalar * explore
        if score == bestscore:
            bestchildren.append(last_move)
        if score > bestscore:
            bestchildren = [last_move]
            bestscore = score
        #print(score, bestchildren, reward, visits)
    next_state = node.state.next_state(move = random.choice(bestchildren)[0])
    for c in node.children:
        if c.state == next_state:
            return c
    raise Exception("Exists no child with the same last move: %s in node: %s"%(next_state.moves[-1], node))

def DEFAULTPOLICY(state, sim_level):
    step = 0
    while state.terminal == False and step < sim_level:
        state = state.next_state()
        step += 1
    return state.reward()

def BACKUP(node, reward):
    while node != None:
        node.visits += 1
        node.reward += reward
        node = node.parent
    return
import time

def main_loop(initial_seed, sim_seed, plan_level, args):
    random.seed(sim_seed)
    current_node = Node(State(initial_seed = initial_seed))
    # print(f"\n{current_node.state.env.state_string()}\n") 
    for l in range(args.levels):
        current_node = UCTSEARCH(args.num_sims, args.sim_level, plan_level, current_node)
        # print(f"Level {l+1}, Reward: {current_node.state.reward()}, Moves: {current_node.state.moves}")
        # print("Snake:", current_node.state.env.snake)
        # for i, c in enumerate(current_node.children):
        #     print(i, c)
        # print(f"\n{current_node.state.env.state_string()}\n")
        # print("-----------------------------------------")
        if current_node.children == []:
            return {
                "begin_seed": initial_seed // 1000 * 1000,
                "plan_level": plan_level,
                "sim_seed": sim_seed,
                "initial_seed": initial_seed,
                "level": l,
                "reward": current_node.state.reward(),
                "final_state": current_node.state.env.state_string(),
            }
    return "Completed all levels without reaching terminal state."
def jobs_to_schedule():
    # sim_level = 10
    # num_sims = 200
    # levels = 100
    tasks = [f"{begin_seed}-{plan_level}" for begin_seed in [5000] for plan_level in [1, 2, 3]]
    instance = []
    if not os.path.exists("logs-mcts"):
        os.makedirs("logs-mcts")
    for task in tasks:
        begin_seed, plan_level = map(int, task.split('-'))
        with open(f"logs-mcts/{begin_seed}_{plan_level}.log", 'w') as f:
            f.write(f"Starting MCTS with num_sims={200}, levels={100}, sim_level={10}, begin_seed={begin_seed}, plan_level={plan_level}\n")
            f.write("Initial Seed, Reward, Level\n")
        for initial_seed in [begin_seed + i for i in range(10)]:
            for sim_seed in [42 + _ for _ in range(3)]:
                instance.append((initial_seed, sim_seed, plan_level))
    assert len(instance) == 90, "Expected 270 instances to process, got %d" % len(instance)
    seed_result = {}
    result = {}
    with ThreadPoolExecutor(max_workers=min(len(instance), 256)) as executor:
        args = argparse.Namespace(
            num_sims=200,
            levels=100,
            sim_level=10,
        )
        futures = [
            executor.submit(
                main_loop, initial_seed, sim_seed, plan_level, args
            )
            for initial_seed, sim_seed, plan_level in instance
        ]
        results = []
        for idx, future in enumerate(as_completed(futures), 1):
            ret = future.result()
            results.append(ret)
            print(f"Progress: {idx}/{len(futures)}")
            with open(f"logs-mcts/{ret['begin_seed']}_{ret['plan_level']}.log", 'a') as f:
                f.write(f"Initial Seed: {ret['initial_seed']}, Reward: {ret['reward']}, Level: {ret['level']}\n")
                f.write(f"Final State:\n{ret['final_state']}\n")
                f.write("---------------------------------------\n")
            if (ret["begin_seed"], ret["plan_level"], ret["initial_seed"]) not in seed_result:
                seed_result[(ret["begin_seed"], ret["plan_level"], ret["initial_seed"])] = []
            seed_result[(ret["begin_seed"], ret["plan_level"], ret["initial_seed"])].append(ret["reward"])
            if (ret["begin_seed"], ret["plan_level"]) not in result:
                result[(ret["begin_seed"], ret["plan_level"])] = []
            result[(ret["begin_seed"], ret["plan_level"])].append(ret["reward"])

        for (begin_seed, plan_level, initial_seed), rewards in seed_result.items():
            mean_reward = sum(rewards) / len(rewards)
            std_reward = math.sqrt(sum([(r - mean_reward) ** 2 for r in rewards]) / len(rewards))
            with open(f"logs-mcts/{begin_seed}_{plan_level}.log", 'a') as f:
                f.write(f"Seed: {initial_seed}, Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}\n")
        for (begin_seed, plan_level), rewards in result.items():
            mean_reward = sum(rewards) / len(rewards)
            std_reward = math.sqrt(sum([(r - mean_reward) ** 2 for r in rewards]) / len(rewards))
            with open(f"logs-mcts/{begin_seed}_{plan_level}.log", 'a') as f:
                f.write(f"Seed: {begin_seed}, Plan Level: {plan_level}, Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}\n")
jobs_to_schedule()
# if __name__=="__main__":
#     parser = argparse.ArgumentParser(description='MCTS research code')
#     parser.add_argument('--num_sims', action="store", required=True, type=int)
#     parser.add_argument('--levels', action="store", required=True, type=int)
#     parser.add_argument('--sim_level', action="store", required=False, type=int, default=10)
#     parser.add_argument('--begin_seed', action="store", required=False, type=int, default=1000)
#     parser.add_argument('--plan_level', action="store", required=False, type=int, default=1)
#     args=parser.parse_args()n_level
#     reward = []
#     start_time = time.time()
#     if not os.path.exists("logs-mcts-debug"):
#         os.makedirs("logs-mcts-debug")
#     seed_reward = {}
#     reward = []
#     with open(f"logs-mcts-debug/{args.begin_seed}_{args.plan_level}.log", 'w') as f:
#         f.write(f"Starting MCTS with num_sims={args.num_sims}, levels={args.levels}, sim_level={args.plan_level}, begin_seed={args.begin_seed}\n")
#         f.write("Initial Seed, Reward, Level\n")
#     with ThreadPoolExecutor(max_workers=1) as executor:
#             futures = [
#                 executor.submit(
#                     main_loop, initial_seed, sim_seed, args
#                 )
#                 for initial_seed in [args.begin_seed + i for i in range(1)] for sim_seed in [42 + _ for _ in range(1)]
#             ]
#             results = []
#             total = len(futures)
#             for idx, future in enumerate(as_completed(futures), 1):
#                 result = future.result()
#                 results.append(result)
#                 # with open(f"logs-mcts-debug/{args.begin_seed}_{args.plan_level}.log", 'a') as f:
#                 #     f.write(f"Initial Seed: {result['initial_seed']}, Reward: {result['reward']}, Level: {result['level']}\n")
#                 #     f.write(f"Final State:\n{result['final_state']}\n")
#                 #     f.write("---------------------------------------\n")
#                 if result["initial_seed"] not in seed_reward:
#                     seed_reward[result["initial_seed"]] = []
#                 seed_reward[result["initial_seed"]].append(result["reward"])
#                 # print(f"Progress: {idx}/{total} ({idx/total*100:.2f}%)")
#                 # print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
#     with open(f"logs-mcts-debug/{args.begin_seed}_{args.plan_level}.log", 'a') as f:
#         for seed, rewards in seed_reward.items():
#             mean_reward = sum(rewards) / len(rewards)
#             std_reward = math.sqrt(sum([(r - mean_reward) ** 2 for r in rewards]) / len(rewards))
#             f.write(f"Seed: {seed}, Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}\n")
#             reward.append(mean_reward)
#         f.write("Mean Total Reward: %.2f\n" % (sum(reward) / len(reward)))
#         f.write("Std Total Reward: %.2f\n" % (math.sqrt(sum([(r - sum(reward) / len(reward)) ** 2 for r in reward]) / len(reward))))
    