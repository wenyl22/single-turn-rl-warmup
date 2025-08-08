import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from minatar.environment import Environment
from minatar.environments.chess2 import Env
from prompts.chess2 import GAME_STATE
import chess

def llm_state_builder(env: Env, time: float, opponent_time: float) -> dict:
    action_history = "No moves made yet"
    lst = env.action_history[env.player_idx]
    if len(lst) > 3:
        action_history = lst[-3:]
    elif len(lst) > 0:
        action_history = lst
    oppoent_action_history = "No moves made yet"
    lst = env.action_history[1 - env.player_idx]
    if len(lst) > 3:
        oppoent_action_history = lst[-3:]
    elif len(lst) > 0:
        oppoent_action_history = lst
    state_for_llm = {
        'color': 'white' if env.player_idx == 0 else 'black',
        'fen': env.board.fen(),
        'time': time,
        'opponent_time': opponent_time,
        'action_history': action_history,
        'opponent_action_history': oppoent_action_history,
        'legal_moves': [move.uci() for move in env.board.legal_moves],
    }
    return state_for_llm
def state_to_description(state_for_llm, scratch_pad=None, fast = False):
    description = GAME_STATE.format(
        color = state_for_llm['color'],
        fen = state_for_llm['fen'],
        time = state_for_llm['time'],
        action_history = state_for_llm['action_history'],
        opponent_time = state_for_llm['opponent_time'],
        opponent_action_history = state_for_llm['opponent_action_history'],
        legal_moves = state_for_llm['legal_moves'],
    )
    if scratch_pad is not None:
        lines = scratch_pad.split('\n')
        for line in lines:
            description += f"> {line.strip()}\n"
    return description
        