SLOW_AGENT_PROMPT = """
## *Blitz Chess Challenge**  
You are playing in a chess game with 10|0 time constraint.  

### **Game Rules**
1. **Move Options**: Output your move in **UCI notation** (e.g., `e2e4`, `g1f3`). In this format, each move is represented by the starting square and ending square of the piece being moved, except for promotion, which is indicated by appending the promotion piece (e.g., `e7e8q` for promoting a pawn to a queen).
2. **Game End Conditions**:  
    - **Win/Loss**: The player who checkmates the opponent wins, or the player who first runs out of time loses.
    - **Automatic Draw**: Stalement, insufficient material, fivefold repetition, or 75-move rule(You don't need to check these conditions, just play until the game ends).
    - **NO resignations or draw offers allowed**—play until forced termination.  
---
### **Game State**  
You will receive:  
1. **Your Color**: It is guaranteed that you are the color that needs to make the next move.
2. **Board State**: FEN string (includes <board_state> <turn> <castling_rights> <en_passant_target_square> <halfmove_clock> <fullmove_number>). 
3. **Action History**: Your and your opponent's last moves(up to 3 moves).
4. **Time Left**: Your and your opponent's remaining time in seconds.
5. **Legal Moves**: List of legal moves available to you.
---
### **Your Task**:
Your ultimate goal is to win the game.
"""
FAST_AGENT_PROMPT = """
## *Blitz Chess Challenge**  
You are playing in a chess game with 10|0 time constraint.  

### **Game Rules**  
1. **Reasoning Trace**:  
   - You may receive **optional strategic advice** from a wise model.  
   - This thinking trace can be **incomplete or incorrect**—use it as a reference or ignore it.  
2. **Move Options**:  
   - **Make a move**: Output your move in **UCI notation** (e.g., `e2e4`, `g1f3`). In this format, each move is represented by the starting square and ending square of the piece being moved, except for promotion, which is indicated by appending the promotion piece (e.g., `e7e8q` for promoting a pawn to a queen).
   - **Wait**: Output `"wait"` to hold for a better reasoning trace. But remember there is time constraint and wait takes time too.  
3. **Game End Conditions**:  
    - **Win/Loss**: The player who checkmates the opponent wins, or the player who first runs out of time loses.
    - **Automatic Draw**: Stalement, insufficient material, fivefold repetition, or 75-move rule(You don't need to check these conditions, just play until the game ends).
    - **NO resignations or draw offers allowed**—play until forced termination.  
---
### **Game State**  
You will receive:  
1. **Your Color**: It is guaranteed that you are the color that needs to make the next move.
2. **Board State**: FEN string (includes <board_state> <turn> <castling_rights> <en_passant_target_square> <halfmove_clock> <fullmove_number>). 
3. **Action History**: Your and your opponent's last moves(up to 3 moves).
4. **Time Left**: Your and your opponent's remaining time in seconds.
5. **Legal Moves**: List of legal moves available to you.
---
### Your Task:
Your ultimate goal is to win the game.

**Answer Format:**
\\boxed{
your move
}

where your move is a legal UCI chess move or `wait` to hold for a better reasoning trace.

### Current State:
"""
DEFAULT_ACTION = "wait"

ACTION_FORMAT_PROMPT = """
**Answer Format:**
\\boxed{
your move
}
---
### Current State:
"""
CONCLUSION_FORMAT_PROMPT = """
**Answer Format:**

**(1) Action:** your move

**(2) Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 
---
### Current State:
"""

GAME_STATE = """
Your Color: 
{color}

Current Board State (FEN): 
{fen}

Your Information:
    - Time Left: {time} seconds
    - Action History: {action_history}

Opponent's Information:
    - Time Left: {opponent_time} seconds
    - Action History: {opponent_action_history}

Legal Moves: {legal_moves}
"""
