
from envs.minatar.environment import Environment
from envs.minatar.environments.airraid import Env

seed_mapping = {0: (60, 465), 1: (183, 460), 2: (245, 444), 3: (592, 440), 4: (696, 476), 5: (794, 513), 6: (945, 478), 7: (987, 424)}

def setup_env(seed, difficulty):
    env = Environment('airraid', sticky_action_prob=0)
    env.seed(seed_mapping[seed][0])
    env.reset()
    return env, seed_mapping[seed]

def summarize(seed, difficulty, env):
    print(f"Seed {seed} - {seed_mapping[seed]}: game_turn: {env.env.game_turn}, reward: {env.env.reward}")
    return False

def llm_state_builder(env: Env):
    player_states = env.pos
    reward_states = []
    for (x, y, speed, reward) in env.space_ships:
        if y > 0:
            reward_states.append((x, y, speed, reward))
    state_for_llm = {
        'turn': env.game_turn,
        'player_states': player_states,
        'reward_states': reward_states,
    }
    return state_for_llm

def state_to_description(state_for_llm, scratch_pad = None):
    description = f"**Current Turn:** \( t_0 = {state_for_llm['turn']} \)\n"
    description += f"*Player Position:** \(({state_for_llm['player_states']}, 0)\) \n"
    description += f"**Reward Value, Position and Speed:**\n"
    description += \
"""
|Reward Value\( r \) | Position \((x, y)\) | Speed \( s \) |
|----------------------|-------------------------|-----------------|\n"""
    for (x, y, speed, reward) in state_for_llm['reward_states']:
        description += f"| {reward} | \({x}, {y}\) | {speed} |\n"
    if scratch_pad is not None:
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description
