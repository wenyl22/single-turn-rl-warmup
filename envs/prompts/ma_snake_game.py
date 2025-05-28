LLM_SYSTEM_PROMPT = """Please think step by step and put your final answer within \\boxed{}."""

GAME_PROMPT = """You are playing a classic Snake game. Your goal is to control the snake to eat food and survive as long as possible.

## Game Rules

* Each time the snake eats food, it grows longer by one segment.
* The snake cannot run into walls or its own body.
* The snake moves one step at a time, in one of four directions: up, down, left, or right.
* The snake cannot reverse direction (i.e., if it is moving up, it cannot move down in the next step).

## Map Representation

The game map is represented by a 2D matrix with the following symbols:

* 'a': Position of the snake's head (always 'a')
* 'b'-'z': Other parts of the snake's body in order ('b' is next to head, 'c' follows 'b', etc.)
* Numbers: Food items, with the number indicating the reward value of the food
* `.` : Empty space
* `#` : Wall

## Output Requirements

Please output the direction letters for the snake's head movement for the next several steps (U-up, D-down, L-left, R-right) based on the current state. Remember, each move shifts the snake one step in the chosen direction. When choosing moves, consider the snake's future path to avoid dead ends. Every time the snake eats food, its tail grows by one unit.

## Answer Format

\\boxed{
    Turn 1: action_1(U/D/L/R)
    Turn 2: action_2
    ...
}

"""

GAME_PROMPT_LOW_LEVEL = """
You are playing a classic Snake game. Your goal is to control the snake to eat food and survive as long as possible.

## Game Rules

* Each time the snake eats food, it grows longer by one segment.
* The snake cannot run into walls or its own body.
* The snake moves one step at a time, in one of four directions: up, down, left, or right.
* The snake cannot reverse direction (i.e., if it is moving up, it cannot move down in the next step).

## Map Representation

The game map is represented by a 2D matrix with the following symbols:

* 'a': Position of the snake's head (always 'a')
* 'b'-'z': Other parts of the snake's body in order ('b' is next to head, 'c' follows 'b', etc.)
* Numbers: Food items, with the number indicating the reward value of the food
* `.` : Empty space
* `#` : Wall

Besides a map, you probably also receive a plan advice(or not), which is a sequence of advised actions \(\{a_{i}^\text{adv}\}_{i=1}^{H}\) (horizon \( H \)), where \( a_{i}^\text{adv} \in \{L, R, U, D\} \). This is only a reference which may be neither safe nor optimal. You can choose to follow the advice and output the first action \( a_{1} = a_{1}^\text{adv} \), or choose your own action.

## Output Requirements

Please output the direction letters for the snake's head movement for the next one step (U-up, D-down, L-left, R-right) based on the current state. Remember, each move shifts the snake one step in the chosen direction. When choosing moves, consider the snake's future path to avoid dead ends. Every time the snake eats food, its tail grows by one unit.


## Answer Format

\\boxed{action(U/D/L/R)}

"""