LLM_SYSTEM_PROMPT = '''Please reason step by step and put your final answer within \\boxed{}.'''

LLM_BASE_PROMPT = """
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

## **Game State Representation**  
Each turn, you receive the **current game state**, which includes:  
- Your position: (x = 0, y).  
- Car information for each freeway (y = 1 to 8). Each may have zero, one or multiple cars, each represented by:
  - Head position (front of the car).  
  - Tail position (back of the car).  
  - Direction (left or right).  
  - Speed (how many x-units the car moves per turn).
"""


SUPERVISOR_PORMPT = """
## **Role**: 

You are the **Supervisor Agent** in the Freeway game.  
Your **only task** is to select which agent should act **this turn** based on the game state.  

### **Agents Available**:  
1. **Plan Agent**: Creates/updates the strategy, and write on the plan scratch pad. Follow plan agent will be immediately called after scratch pad update.
2. **Follow Plan Agent**: Executes the current plan.  
3. **React Agent**: Handles emergencies.  

### **Possible Decision Rules**:
1. **Is there an immediate collision risk by following the plan?**
   - **YES**: Choose **React Agent** (e.g., a car suddenly appears and will hit the player next turn).  
   - **NO**: Proceed to step 2.  
2. **Does the scratch pad have a valid, up-to-date plan?**  
   - **YES**: Choose **Follow Plan Agent**.  
   - **NO**: Choose **Plan Agent**.  

### **Output Format**:  
\boxed{Plan Agent | Follow Plan Agent | React Agent}  

## **Current Game State**:  
"""
# Plan agent, plan and modify the plan scratch pad
PLAN_PROMPT = """
## **Your Task**:
Analyze the current game state and create a plan for crossing the road without colliding with cars. Avoiding collisions is your top priority, but also try to reach the goal in the fewest turns possible. Due to token constraint, you can plan for less steps ** if the game state is complex **.

## **Answer Format**:
\boxed{
Turn 1: action1
Turn 2: action2
...
}

## **Current Game State**:
"""

# Follow plan agent, follow the plan scratch pad
FOLLOW_PLAN_PROMPT = """
## **Your Task**: 
Read the plan on the scratch pad and act accordingly. You are short in time, so do not overthink or plan for the future.

## **Answer Format**:  
\boxed{selected action}
## **Current Game State**:
"""
# React agent, react to the game state
REACT_PROMPT = """
## **Your Task**: 
Analyze the current game state and take immediate action to avoid a collision. You don't have to follow the content on the scratch bad as it may be out-dated. You are short in time, so do not overthink or plan for the future.

## **Answer Format**:
\boxed{selected action} 
"""

STAY_COMPLETION = """Stay in the same freeway"""


