SLOW_AGENT_PROMPT = """
### **Game Rules**
You are playing as **White** in a fast-paced chess game with a unique time constraint.  
1. **Move Options**: Output your move in **UCI notation** (e.g., `e2e4`, `g1f3`). In this format, each move is represented by the starting square and ending square of the piece being moved, except for promotion, which is indicated by appending the promotion piece (e.g., `e7e8q` for promoting a pawn to a queen).
2. **Game End Conditions**:  
    - **Win/Loss**: Checkmate the opponent or be checkmated.
    - **Automatic Draw**: Stalement, insufficient material, fivefold repetition, or 75-move rule(You don't need to check these conditions, just play until the game ends).
    - **NO resignations or draw offers allowed**—play until forced termination.  
---
### **Game State**  
You will receive:  
1. **Board State**: FEN string (includes <board_state> <turn> <castling_rights> <en_passant_target_square> <halfmove_clock> <fullmove_number>)
2. **Action History**: Your and your opponent's last moves(up to 3 moves).  
---
### **Your Task**:
Your ultimate goal is to win the game.
"""
FAST_AGENT_PROMPT = """
## **Speed Chess Challenge: Limited Time Bubbles**  
You are playing as **White** in a fast-paced chess game with a unique time constraint.  

### **Game Rules**  
1. **Time Bubbles**:  
   - You start with a few time bubbles.  
   - Each move consumes **at least 1 time bubble**.  
   - Your opponent's moves do **not** consume your time bubbles.  
2. **Move Options**:  
   - **Make a move**: Output your move in **UCI notation** (e.g., `e2e4`, `g1f3`). In this format, each move is represented by the starting square and ending square of the piece being moved, except for promotion, which is indicated by appending the promotion piece (e.g., `e7e8q` for promoting a pawn to a queen).
   - **Wait**: Output `"wait"` to hold for a better reasoning trace (consumes **1 time bubble**).  
3. **Reasoning Trace**:  
   - You may receive **optional strategic advice** from a wise model. This model thinks about the exact same game state as you, but it has no awareness of how many time bubbles you have left.  
   - This trace can be **incomplete or incorrect**—use it as a reference or ignore it.  
4. **Game End Conditions**:  
    - **Win/Loss**: Checkmate the opponent or be checkmated.
    - **Draw**: One of the opponent has no time bubbles left.
    - **Automatic Draw**: Stalement, insufficient material, fivefold repetition, or 75-move rule(You don't need to check these conditions, just play until the game ends).
    - **NO resignations or draw offers allowed**—play until forced termination.  
---
### **Game State**  
You will receive:  
1. **Board State**: FEN string (includes <board_state> <turn> <castling_rights> <en_passant_target_square> <halfmove_clock> <fullmove_number>)
2. **Action History**: Your and your opponent's last moves(up to 3 moves).  
3. **Time Bubbles Left**: Your and your opponent's remaining time bubbles.
4. **Reasoning Trace**: Strategic suggestions (may be incomplete or not available).  
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
ALL_ACTIONS = ""
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

FAST_GAME_STATE = """
Current Board State (FEN): 
{fen}

Your Information:
    - Time Bubbles Left: {time_bubbles}
    - Action History: {action_history}

Opponent's Information:
    - Time Bubbles Left: {opponent_time_bubbles}
    - Action History: {opponent_action_history}

Legal Moves: {legal_moves}
"""

SLOW_GAME_STATE = """
Current Board State (FEN):
{fen}

Your Information:
    - Action History: {action_history}

Opponent's Information:
    - Action History: {opponent_action_history}

Legal Moves: {legal_moves}
"""