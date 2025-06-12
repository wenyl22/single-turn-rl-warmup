from envs.minatar.environment import Environment
from envs.minatar.environments.snake import Env
from copy import deepcopy

VLLM_client = None 

def setup_env(seed, difficulty):
    env = Environment("snake", sticky_action_prob=0)
    env.seed(seed + 1000)
    env.reset()
    return env, seed + 1000

def summarize(seed, difficulty, thread_id, env, client):
    print(f"Seed {seed} - {env.env.game_turn} turns, reward: {env.env.reward}")
    return 

def llm_state_builder(env: Env):
    snake = deepcopy(env.snake[::-1])
    foods = deepcopy(env.food)
    return {
        "map": env.state_string(), 
        "snake_dir": env.dir,
        "foods": foods,
        "snake": snake
    }

def state_to_description(state_for_llm, scratch_pad = None):
    description = """## Current game state\n"""
    description += f"""**Snake Positions**:{state_for_llm['snake']}\n**Snake Head Direction**: {state_for_llm['snake_dir']}\n"""
    description += f"**Food Positions, Life Span and Value**:\n"
    for (x, y, value, life_span) in state_for_llm['foods']:
        description += f"\t- ({x}, {y}, {life_span}, {value})\n"
    if scratch_pad is not None:
        description += f"**Plan Advice**: {",".join(scratch_pad)}\n"
    return description
