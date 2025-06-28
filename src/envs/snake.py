from envs.minatar.environment import Environment
from envs.minatar.environments.snake import Env
from copy import deepcopy
seed_mapping = {
    'E': {0: 1000, 1: 1001, 2: 1002, 3: 1003, 4: 1004, 5: 1005, 6: 1006, 7: 1007},
    'M': {0: 2000, 1: 2001, 2: 2002, 3: 2003, 4: 2004, 5: 2005, 6: 2006, 7: 2007},
    'H': {0: 3000, 1: 3001, 2: 3002, 3: 3003, 4: 3004, 5: 3005, 6: 3006, 7: 3007},
}
def setup_env(seed, difficulty):
    env = Environment("snake", sticky_action_prob=0)
    env.seed(seed_mapping[difficulty][seed])
    env.reset()
    return env, seed_mapping[difficulty][seed]

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
        "snake": snake,
        "size": env.B
    }

def state_to_description(state_for_llm, scratch_pad = None, fast = False):
    description = f"**Current Turn**: \( t_{0 if fast else 1} = {state_for_llm['turn']} \)\n"
    description += f"**Cells occupied by walls**: `x=0`/`x={state_for_llm['size'] - 1}` or `y=0`/`y={state_for_llm['size'] - 1}`.\n"
    description += f"**Snake Positions**:{state_for_llm['snake']}\n**Snake Head Direction**: {state_for_llm['snake_dir']}\n"
    description += f"**Food Positions, Life Span and Value**:\n"

    for (x, y, life_span, value) in state_for_llm['foods']:
        description += f"\t- ({x}, {y}, {life_span}, {value})\n"
    if scratch_pad is not None:
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description
