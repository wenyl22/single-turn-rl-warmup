import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from envs.minatar.environment import Environment
from freeway_seed_generation import bfs
parser = argparse.ArgumentParser(description="Visualize the seed of a model.")
parser.add_argument("--s", type=int, default=0, help="The seed to visualize.")
args = parser.parse_args()
seed = args.s
env = Environment('freeway', sticky_action_prob=0)
env.seed(seed)
env.reset()
best_action = bfs(env, max_steps=100)
env.seed(seed)
env.reset()
with open(f"seed_{seed}.txt", "w") as f:
    print (f"Seed {seed} best action: {best_action}")
    for a in best_action:
        f.write("--" * 20 + "\n")
        f.write(env.env.state_string() + "\n")
        env.act(a)