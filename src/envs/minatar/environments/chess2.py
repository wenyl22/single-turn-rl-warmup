import chess
import chess.engine
import numpy as np

from typing import Optional, Tuple

class Env:
    def __init__(self, ramping=None):
        self.random = np.random.RandomState()
        self.seed = 42
        self.time_limit = 1.0
    def reset(self):
        self.board = chess.Board()
        self.action_history = [[], []]
        self.reward = 0
        self.terminal = False
        self.action_made = True
        self.player_idx = 0

    def _parse_action(self, action_str: str) -> Optional[chess.Move]:
        try:
            action_str = action_str.strip()
            if len(action_str) >= 2:
                try:
                    move = self.board.parse_san(action_str)
                    return move
                except:
                    pass
            action_str = action_str.lower()
            if len(action_str) == 4 or len(action_str) == 5:
                try:
                    move = chess.Move.from_uci(action_str)
                    if move in self.board.legal_moves:
                        return move
                except:
                    pass
            return None
        except:
            return None

    def act(self, player_move) -> Tuple[float, bool]:
        assert not self.terminal, "Game is already over."
        #player_move = self._parse_action(action_str)
        assert player_move and player_move in self.board.legal_moves
        self.action_made = True
        self.board.push(player_move)
        self.action_history[self.player_idx].append(player_move.uci())
        if self.board.is_game_over():
            self.r = self._calculate_final_reward()
        self.reward += self.r
        self.player_idx = 1 - self.player_idx
        return self.r, self.terminal

    def _calculate_final_reward(self):
        r = 0
        result = self.board.result()
        if result == "1-0":  # player wins
            r += 100
        elif result == "0-1":  # Stockfish wins
            r -= 100
        else: 
            r += 10
        self.terminal = True 
        return r

    def state_string(self) -> str:
        board_str = str(self.board)
        return board_str

    def get_legal_moves(self) -> list:
        return [move.uci() for move in self.board.legal_moves]

    def __del__(self):
        if hasattr(self, 'engine') and self.engine:
            try:
                self.engine.quit()
            except:
                pass