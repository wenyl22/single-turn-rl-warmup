from envs.minatar.environment import Environment
from envs.minatar.environments.snake import Env
from copy import deepcopy

def setup_env(seed, difficulty):
    env = Environment("snake", sticky_action_prob=0)
    env.seed(seed + 1000)
    env.reset()
    return env, seed + 1000

def summarize(seed, difficulty, env):
    print(f"Seed {seed} - {env.env.game_turn} turns, reward: {env.env.reward}")
    return False

def llm_state_builder(env: Env):
    snake = deepcopy(env.snake[::-1])
    foods = []
    for (x, y) in env.food:
        l, v = env.food_attributes[x][y]
        foods.append((x, y, l, v))
    return {
        "turn": env.game_turn,
        "snake_dir": env.dir,
        "foods": foods,
        "snake": snake
    }

def state_to_description(state_for_llm, scratch_pad = None):
    description = f"**Current Turn**: \( t_0 = {state_for_llm['turn']} \)\n"
    description += f"""**Snake Positions**:{state_for_llm['snake']}\n**Snake Head Direction**: {state_for_llm['snake_dir']}\n"""
    description += f"**Food Positions, Life Span and Value**:\n"
    for (x, y, life_span, value) in state_for_llm['foods']:
        description += f"\t- ({x}, {y}, {life_span}, {value})\n"
    if scratch_pad is not None:
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description
