import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from minatar.environment import Environment
from minatar.environments.chess import Env
from prompts.chess import FAST_GAME_STATE, SLOW_GAME_STATE
import chess

def setup_env(seed, difficulty):
    env = Environment('chess', sticky_action_prob=0)
    env.env.difficulty = 1 if difficulty == 'E' else 3 if difficulty == 'M' else 4
    env.seed(seed)
    env.reset()
    return env, seed

def llm_state_builder(env: Env):
    state_for_llm = {
        'fen': env.board.fen(),
        'time_bubbles': env.time_bubbles,
        'opponent_time_bubbles': env.opponent_time_bubbles,
        'action_history': env.action_history[-3:] if len(env.action_history) > 3 else env.action_history if env.action_history else "No moves yet",
        'opponent_action_history': env.opponent_action_history[-3:] if len(env.opponent_action_history) > 3 else env.opponent_action_history if env.opponent_action_history else "No moves yet",
        'legal_moves': [move.uci() for move in env.board.legal_moves],
    }
    return state_for_llm
def state_to_description(state_for_llm, scratch_pad=None, fast = False):
    fen = state_for_llm['fen']
    if fast:
        description = FAST_GAME_STATE.format(
            fen=fen,
            time_bubbles=state_for_llm['time_bubbles'],
            action_history=state_for_llm['action_history'],
            opponent_time_bubbles=state_for_llm['opponent_time_bubbles'],
            opponent_action_history=state_for_llm['opponent_action_history'],
            legal_moves=state_for_llm['legal_moves'],
        )
    else:
        description = SLOW_GAME_STATE.format(
            fen=fen,
            action_history=state_for_llm['action_history'],
            opponent_action_history=state_for_llm['opponent_action_history'],
            legal_moves=state_for_llm['legal_moves'],
        )
    if scratch_pad is not None:
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description
        
def summarize(seed, difficulty, env):
    print(f"Seed {seed} - {difficulty} turn: {env.env.game_turn}, reward: {env.env.reward}")
    return env.env.action_made  
    # Return True if no action was made (slow keeps thinking), indicating a wait or no move situation
    # Return False if an action was made (slow drops previous reasoning)