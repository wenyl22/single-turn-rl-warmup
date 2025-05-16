LLM_SYSTEM_PROMPT = '''Please reason step by step and put your final answer within \\boxed{}.'''

LLM_GAME_PROMPT = """
## **Freeway Game Guide**  

Imagine a 2D grid where:  
- The vertical axis (y) represents different freeways, numbered from `0` to `9`.  
- The horizontal axis (x) represents positions along each freeway.  

Your goal is to reach (x = 0, y = 9) from (x = 0, y = 0) using actions U (Up, to a higher freeway, y += 1), D (Down, to a lower freeway, y -= 1), or S (Stay, y unchanged) in fewest turns, while avoiding cars on the freeways. 

# **Rules**  

## **1. Freeways & Cars**  
Each freeway (from y = 1 to y = 8) may have cars moving left or right.  
- **Cars move at a fixed speed per turn** (e.g., a speed of 3 means the car moves 3 units in the x-direction each turn).  
- **Each car has a span**, which is the range of x-values it occupies.
    - If the car is moving **left**, its span is defined as: **[head position, tail position]**
    - If the car is moving **right**, its span is defined as: **[tail position, head position]**

**Example:**  
A car on Freeway 2 with a **head position at x = 18** and a **tail position at x = 29** moves **left** at a **speed = 6**.  
    - After **one turn**, its span will be **[head: 12, tail: 23]**.  
    - After **two turns**, its span will be **[head: 6, tail: 17]**.  

## **2. Collisions**  
Collision happens if after the player's move, its position overlaps with a car's span, which should be avoided. Collision check happens only after the player and cars move for a complete turn, so don't worry about the state within the turn.

## **Your Task**:
Analyze the current game state and create a plan for crossing the road without colliding with cars. Avoiding collisions is your top priority, but also try to reach the goal in the fewest turns possible. Due to token constraint, you can plan for less steps ** if the game state is complex **.

## **Answer Format**:
\\boxed{
Turn 1: action1
Turn 2: action2
...
}
"""


STAY_COMPLETION = """Stay in the same freeway"""

