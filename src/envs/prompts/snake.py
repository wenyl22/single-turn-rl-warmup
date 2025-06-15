LLM_SYSTEM_PROMPT = """Please think step by step and put your final answer within \\boxed{}."""

SLOW_AGENT_PROMPT = """
You are an AI playing Snake on an 8×8 grid. Control the snake to maximize survival time and score by eating positive food while avoiding risks.

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
- Body collision (head touches any body segment)
- Wall collision (grid borders)
  - Walls: `x=0`/`x=7` or `y=0`/`y=7`
  - Corner collisions included

## State Input Format

**Snake Positions**: [(x0,y0), (x1,y1), ...] (head first, body segments follow)
**Snake Head Direction**: : "U/D/L/R"               
**Food Positions, Value and Life Span**:
- ( (x1,y1), value1, life_span1 )                  
- ( (x2,y2), value2, life_span2 )
}

## Answer Format

Your answer **must** include both of the following, clearly separated:

**1. Action Sequence (in order):**

\\boxed{
    Turn 1: action_1
    Turn 2: action_2
    ...
}

Where each action \(action_t \in \{\text{U (up)},\ \text{D (down)},\ \text{L (left)},\ \text{R (right)}\}\).

**2. Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 


"""


FAST_AGENT_PROMPT= """
You are an AI playing Snake on an 8×8 grid. Control the snake to maximize survival time and score by eating positive food while avoiding risks. As a **non-thinking executor**, your task is to output **only the next immediate action** based on:
1. Current game state
2. Thinking Model's past plan
3. Deviation analysis between past plan and current reality

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
- Body collision (head touches any body segment)
- Wall collision (grid borders)
  - Walls: `x=0`/`x=7` or `y=0`/`y=7`
  - Corner collisions included

## State Input Format

**Snake Positions**: [(x0,y0), (x1,y1), ...] (head first, body segments follow)
**Snake Head Direction**: : "U/D/L/R"               
**Food Positions, Value and Life Span**:
- ( (x1,y1), value1, life_span1 )                  
- ( (x2,y2), value2, life_span2 )
}

## Answer Format

\\boxed{action(U/D/L/R)}


"""

# FAST_AGENT_PROMPT = """
# You are playing a classic Snake game. Your goal is to control the snake to eat food and survive as long as possible.

# ## Game Rules

# * Each time the snake eats food with +1 value, it grows longer by one segment and gets a reward of +1.
# * Each time the snake eats food with -1 value, its length does not change and gets a reward of -1.
# * Each food has a life span, which means how many turns(including the current one) it can stay on the board before disappearing.
# * The snake cannot run into walls or its own body. There are 4 walls surrounding the game area, which are:
#     * (0, 0) to (0, 7) - Left wall
#     * (0, 0) to (7, 0) - Bottom wall
#     * (7, 0) to (7, 7) - Right wall
#     * (0, 7) to (7, 7) - Top wall
# The wall takes up the whole row or column, so the snake cannot move to these coordinates.
# * The snake moves one step at a time, in one of four directions: up(y += 1), down(y -= 1), left(x -= 1), or right(x += 1).
# * The snake cannot reverse direction (i.e., if it is moving up, it cannot move down in the next step).

# ## State Representation

# The game state is described with following details:

# * Snake positions: The snake is represented by a sequence of coordinates, where the first coordinate is the head of the snake, followed by its body segments. The neighboring segments are adjacent to each other in the order they appear.

# * Snake head direction: The direction of the snake's head is indicated by a letter ('U', 'D', 'L', 'R') corresponding to the movement direction.

# * Food positions: The food is represented by a list of coordinates and corresponding reward values and life spans.


# Besides a map, you probably also receive a plan advice(or not), which is a sequence of advised actions \(\{a_{i}^\text{adv}\}_{i=1}^{H}\) (horizon \( H \)), where \( a_{i}^\text{adv} \in \{L, R, U, D\} \). This is only a reference which may be neither safe nor optimal. You can choose to follow the advice and output the first action \( a_{1} = a_{1}^\text{adv} \), or choose your own action.

# ## Output Requirements

# Please output the direction letters for the snake's head movement for the next one step (U-up, D-down, L-left, R-right) based on the current state. When choosing move for current turn, also think about the consequence and consider the snake's **future moves** to avoid trapping yourself.

# ## Answer Format

# \\boxed{action(U/D/L/R)}

# """

DEFAULT_ACTION = 'S'

ALL_ACTIONS = 'LRUD'