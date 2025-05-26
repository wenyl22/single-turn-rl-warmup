import argparse
from envs.minatar.environment import Environment
from seed_generation import greedy
parser = argparse.ArgumentParser(description="Visualize the seed of a model.")
parser.add_argument("--s", type=int,nargs='+', default=[0], help="Seed for the environment")
args = parser.parse_args()
seeds = args.s
if args.s == [0]:
    seeds = [375, 978, 1005, 1337, 1800, 1983, 2029, 2675]
seed_list = {}
for i, seed in enumerate(seeds):
    env = Environment('airraid', sticky_action_prob=0)
    env.seed(seed)
    env.reset()
    best_action, _, _ = greedy(env, max_steps=100)
    env.seed(seed)
    env.reset()
    seed_list[i] = (seed, len(best_action), 0)
    reward = 0
    with open(f"seed_{seed}.txt", "w") as f:
        for a in best_action:
            f.write("--" * 20 + "\n")
            f.write(env.env.state_string() + "\n")
            if a == "L":
                act = 1
            elif a == "R":
                act = 3
            else:
                act = 0
            r, t = env.act(act)
            reward += r
            f.write("Action: " + a + " Reward: " + str(reward) + "\n")
print(seed_list)