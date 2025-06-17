LLM_SYSTEM_PROMPT = """Please think step by step and put your final answer within \\boxed{}."""

SLOW_AGENT_PROMPT = """
You are an AI playing Snake on an 8x8 grid. Maximize survival time and score by eating positive food while avoiding risks. **Think 5+ turns ahead** - prioritize long-term survival over immediate rewards.
## Core Rules 
**1. Food Mechanics**
- **Positive Food** 
  - `Reward Value = +1` | `Length of snake +1`
- **Negative Food** 
  - `Penalty Value = -1` | `Length of snake unchanged`
- **Life-span** 
  - Disappears after N turns (N = life_span)
  - Countdown decreases every turn (including current)

**2. Movement Constraints**
- In each turn you can choose to move 1 cell in following directions: U`(x,y+1)`, D`(x,y-1)`, L`(x-1,y)`, R`(x+1,y)`
- **No instant reverse**: Cannot move opposite to current direction on next turn

**3. Deadly Collisions**
- Body collision (head touches any body segment). But note you can still move into the cell occupied by the tail segment, since the tail will also move forward.
- Wall collision (grid borders)
  - Cells occupied by walls: `x=0`/`x=7` or `y=0`/`y=7`.
  - The wall takes up the whole row or column, so the snake cannot move to these coordinates.

## State Input Format
**Current turn**: \(t_1 = some integer\)
**Snake Positions**: [(x0,y0), (x1,y1), ...] (head first, body segments follow)
**Snake Head Direction**: : "U/D/L/R"               
**Food Positions, Value and Life Span**:
- ( (x1,y1), value1, life_span1 )                  
- ( (x2,y2), value2, life_span2 )

"""

ACTION_FORMAT_PROMPT= """
## Answer Format

\\boxed{
Turn \(t_1\): action on turn t_1
Turn \(t_1 + 1\): action on turn t_1 + 1
...
}

Where each action \(action \in \{\text{U (up)},\ \text{D (down)},\ \text{L (left)},\ \text{R (right)}\}\).

## Current State (Turn \(t_1\)):
"""

CONCLUSION_FORMAT_PROMPT = """
## Answer Format

Your answer **must** include both of the following, clearly separated:

**1. Action Sequence (in order):**

\\boxed{
Turn \(t_1\): action on turn t_1
Turn \(t_1 + 1\): action on turn t_1 + 1
...
}

Where each action \(action \in \{ U, D, L, R \}\).

**2. Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 

## Current State (Turn \(t_1\)):
"""

FAST_AGENT_CONCLUSION_PROMPT= """
You are an AI playing Snake on an 8×8 grid. Control the snake to maximize survival time and score by eating positive food while avoiding risks. As a **non-thinking executor**, your task is to decide **the immediate action for the current Turn \(t_0\** based on:
1. Current game state
2. Thinking Model's past plan

Action will apply BEFORE countdown updates.

## Core Rules 
**1. Food Mechanics**
- **Positive Food** 
  - `Reward = +1` | `Length of snake +1`
- **Negative Food** 
  - `Penalty = -1` | `Length of snake unchanged`
- **Life-span** 
  - Disappears after N turns (N = life_span)
  - Countdown decreases every turn (including current)

**2. Movement Constraints**
- Moves 1 cell/turn: U`(x,y+1)`, D`(x,y-1)`, L`(x-1,y)`, R`(x+1,y)`
- **No instant reverse**: Cannot move opposite to current direction on next turn

**3. Deadly Collisions**
- Body collision (head touches any body segment). But note you can still move into the cell occupied by the tail segment, since the tail will also move forward.
- Wall collision (grid borders)
  - Cells occupied by walls: `x=0`/`x=7` or `y=0`/`y=7`.
  - **Walls have*width of 1 cell**, so the snake head cannot move to cells with `x=0`/`x=7` or `y=0`/`y=7`.

## State Input Format
**Current turn**: \(t_0 = some integer\)
**Snake Positions**: [(x0,y0), (x1,y1), ...] (head first, body segments follow)
**Snake Head Direction**: : "U/D/L/R"               
**Food Positions, Value and Life Span**:
- ( (x1,y1), value1, life_span1 )                  
- ( (x2,y2), value2, life_span2 )

## Answer Format

\\boxed{action(U/D/L/R)}

## Current State (Turn \(t_0\)):
"""


FAST_AGENT_ACTION_PROMPT = FAST_AGENT_CONCLUSION_PROMPT

DEFAULT_ACTION = 'S'

ALL_ACTIONS = 'LRUD'