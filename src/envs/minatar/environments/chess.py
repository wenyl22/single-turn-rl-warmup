import chess
import chess.engine
import numpy as np

from typing import Optional, Tuple

class Env:
    def __init__(self, ramping=None):
        self.random = np.random.RandomState()
        self.seed = 42
        self.stockfish_path = "/usr/games/stockfish"
        self.engine = None
        self.time_limit = 1.0
        self.difficulty = 1
    def reset(self):
        self.board = chess.Board()
        self.time_bubbles = 200
        self.opponent_time_bubbles = 200
        self.action_history = []
        self.opponent_action_history = []
        self.reward = 0
        self.game_turn = 0
        self.terminal = False
        self.action_made = True
        self.player_color = chess.WHITE  # Player plays white        
        try:
            if self.engine:
                self.engine.quit()
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({"Skill Level": self.difficulty + 1})
        except Exception as e:
            print(f"Unable to start Stockfish engine: {e}")
            self.engine = None

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

    def _get_prandom_stockfish_move(self) -> Optional[chess.Move]:
        if not self.engine or self.board.is_game_over():
            return None            
        try:
            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                return None
            
            best_moves = []
            best_score = float('-inf') if self.board.turn == chess.WHITE else float('inf')
            
            for move in legal_moves:
                self.board.push(move)
                try:
                    info = self.engine.analyse(self.board, chess.engine.Limit(time = 1.0))
                    score = info["score"].white().score(mate_score=10000)
                    if self.board.turn == chess.WHITE:
                        if score < best_score:
                            best_score = score
                            best_moves = [move]
                        elif score == best_score:
                            best_moves.append(move)
                    else:
                        if score > best_score:
                            best_score = score
                            best_moves = [move]
                        elif score == best_score:
                            best_moves.append(move)
                except Exception as e:
                    print(f"Move analysis error: {e}")
                finally:
                    self.board.pop()
            
            if best_moves:
                return self.random.choice(best_moves)
            else:
                return legal_moves[0]                
        except Exception as e:
            print(f"Stockfish move error: {e}")
            return None
    def act(self, action_str: str) -> Tuple[float, bool]:
        assert not self.terminal, "Game is already over."
        self.r = 0
        self.game_turn += 1
        self.time_bubbles -= 1
        if self.time_bubbles <= 0:
            self.terminal = True
        player_move = self._parse_action(action_str)
        if not player_move or player_move not in self.board.legal_moves:
            action_str = "wait"
        if action_str == "wait":
            self.action_made = False
            return 0, self.terminal

        self.action_made = True
        self.board.push(player_move)
        self.action_history.append(player_move.uci())
        if self.board.is_game_over():
            self.r = self._calculate_final_reward()
            return self.r, self.terminal            

        stockfish_move = self._get_prandom_stockfish_move()
        self.opponent_time_bubbles -= 1
        if stockfish_move:
            self.board.push(stockfish_move)
            self.opponent_action_history.append(stockfish_move.uci())
            if self.board.is_game_over():
                self.r = self._calculate_final_reward()
        return self.reward, self.terminal

    def _calculate_final_reward(self):
        reward = 0
        result = self.board.result()
        if result == "1-0":  # player wins
            reward += 100
        elif result == "0-1":  # Stockfish wins
            reward -= 100
        else: 
            reward += 10
        self.terminal = True 
        self.reward += reward
        return reward

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