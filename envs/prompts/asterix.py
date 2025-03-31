LLM_SYSTEM_PROMPT = '''Please put your final answer within \\boxed{}. Do not include any extra text.'''

LLM_BASE_PROMPT = '''
# **Asterix Game: Optimal Action Selection**  

## **Game Overview**  
You are playing Asterix in a 2D grid. Collect treasures (+1 rewards) and avoid monsters (game ends). Your goal is to maximize your rewards while avoiding monsters.

## **Game Rules**  

### **1. Control**
- You can control the main character moving UP, DOWN, LEFT, RIGHT, or STAY in the same position.

### **2. Entities**  
- Entities in the game are monsters or treasures. 
- They move horizontally (along the x-axis), either left or right.
- At each turn, you move first, and then all entities move if it's their turn. You contact with an object if you are at the same position **before** or **after** its movement.

### **3. Movement Prediction**
- Entities only move **on specific turns**.
- Use their last 4 movement turns to predict their next move.

Example: If a monster moved on turns [3, 5, 7, 9], it likely moves every 2 turns (next: turn 11).

## **Game State Representation**  
Each turn, you receive the **current game state**, including:
- **Current turn number**: t.
- **Your position**: (x, y).  
- **Entity information**:  
  - **Entity type** (monster or treasure).
  - **Position** (x, y).
  - **Direction** (left or right).  
- **Entity movement history**: (t1, t2, t3, t4). This means all treasures and monsters moved one unit forward in their direction on turns t1, t2, t3, and t4. **This can be used to predict whether the entities will move in the next turn**.
- **Available actions**: a subset of the following:
  - Move UP (y -> y + 1).
  - Move DOWN (y -> y - 1).
  - Move LEFT (x -> x - 1).
  - Move RIGHT (x -> x + 1).
  - Stay in the same position.

## **Your Task: Find the Best Move**  
Analyze the game state, predict the entities' next moving turn and select your move from available actions in the current turn.

## **Given Game State:**
'''
STAY_COMPLETION = """Stay at (x = {x}, y = {y})"""