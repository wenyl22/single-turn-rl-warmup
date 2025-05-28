LLM_SYSTEM_PROMPT = """Please think step by step and put your final answer within \\boxed{}."""

GAME_PROMPT = """
You are playing a classic Snake game. Your goal is to control the snake to eat food and survive as long as possible.

## Game Rules

* Each time the snake eats food, it grows longer by one segment.
* The snake cannot run into walls or its own body. There are 4 walls surrounding the game area, which are:
    * (0, 0) to (0, 7) - Top wall
    * (0, 0) to (7, 0) - Left wall
    * (7, 0) to (7, 7) - Bottom wall
    * (0, 7) to (7, 7) - Right wall
* The snake moves one step at a time, in one of four directions: up(y -= 1), down(y += 1), left(x -= 1), or right(x += 1).
* The snake cannot reverse direction (i.e., if it is moving up, it cannot move down in the next step).

## State Representation

The game state is described with following details:
* Snake positions: The snake is represented by a sequence of coordinates, where the first coordinate is the head of the snake, followed by its body segments. The neighboring segments are adjacent to each other in the order they appear.
* Snake head direction: The direction of the snake's head is indicated by a letter ('U', 'D', 'L', 'R') corresponding to the movement direction.
* Food positions: The food is represented by a list of coordinates and corresponding reward values.
## Output Requirements

Please output the direction letters for the snake's head movement for the next **several turns** (U-up, D-down, L-left, R-right) based on the current state. Remember, each move shifts the snake one step in the chosen direction. When choosing moves, consider the snake's future path to avoid dead ends. Every time the snake eats food, its tail grows by one unit.

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
* The snake cannot run into walls or its own body. There are 4 walls surrounding the game area, which are:
    * (0, 0) to (0, 7) - Top wall
    * (0, 0) to (7, 0) - Left wall
    * (7, 0) to (7, 7) - Bottom wall
    * (0, 7) to (7, 7) - Right wall
* The snake moves one step at a time, in one of four directions: up(y -= 1), down(y += 1), left(x -= 1), or right(x += 1).
* The snake cannot reverse direction (i.e., if it is moving up, it cannot move down in the next step).

## State Representation

The game state is described with following details:
* Snake positions: The snake is represented by a sequence of coordinates, where the first coordinate is the head of the snake, followed by its body segments. The neighboring segments are adjacent to each other in the order they appear.
* Snake head direction: The direction of the snake's head is indicated by a letter ('U', 'D', 'L', 'R') corresponding to the movement direction.
* Food positions: The food is represented by a list of coordinates and corresponding reward values.

Besides a map, you probably also receive a plan advice(or not), which is a sequence of advised actions \(\{a_{i}^\text{adv}\}_{i=1}^{H}\) (horizon \( H \)), where \( a_{i}^\text{adv} \in \{L, R, U, D\} \). This is only a reference which may be neither safe nor optimal. You can choose to follow the advice and output the first action \( a_{1} = a_{1}^\text{adv} \), or choose your own action.

## Output Requirements

Please output the direction letters for the snake's head movement for the next one step (U-up, D-down, L-left, R-right) based on the current state. Remember, each move shifts the snake one step in the chosen direction. When choosing moves, consider the snake's future path to avoid dead ends. Every time the snake eats food, its tail grows by one unit.


## Answer Format

\\boxed{action(U/D/L/R)}

"""