LLM_SYSTEM_PROMPT = '''Please put your final answer within \\boxed{}. Do not include any extra text.'''

LLM_BASE_PROMPT = '''
# **Asterix Game: Optimal Action Selection**  

## **Game Overview**  
You are playing "Asterix". Your goal is to move the character on 2D Cartesian plane, collect as many treasures as possible while avoiding monsters. On each turn, you can move one unit in one of the four cardinal directions (up, down, left, right). You can also choose to stay in the same position.

## **Game Mechanics**  

### **1. Entities**  
Entities in the game include monsters and treasures. 
- Entities move horizontally (along the x-axis), either left or right.
- All Entities move one unit in their direction only **on specific turns**, which needs to be **predicted based on their recent movement history**.
- At each turn, you move first, and then all entities move if it's their turn. You contact with an object if you are at the same position **before** or **after** its movement.

**Example:**  
A monster at position (x = 5, y = 3) moves left. You know all treasures and monsters moved one unit in their direction on turns t1, t2, t3, and t4. Then you may predict the monster will move to (x = 4, y = 3) in t5 turn. If you are at (x = 5, y = 3) or (x = 4, y = 3) at the end of t5 turn, you will contact with the monster.

### **2. Rewards and Penalties**
- **Collecting a treasure**: +1 point.
- **Contacting with a monster**: the game ends.

## **Game State Representation**  
Each turn, you receive the **current game state**, which includes:
- **Current turn number**: t.
- **Your position**: (x, y).  
- **Entity information**:  
  - **Entity type** (monster or treasure).
  - **Position** (x, y).
  - **Direction** (left or right).  
- **Entity movement history**: (t1, t2, t3, t4). This means all treasures and monsters moved one unit forward in their direction on turns t1, t2, t3, and t4. **This can be used to predict whether the entities will move in the next turn**.
- **Available actions**: a subset of the following:
  - Move **up** (y -> y + 1).
  - Move **down** (y -> y - 1).
  - Move **left** (x -> x - 1).
  - Move **right** (x -> x + 1).
  - Stay in the same position.

## **Your Task: Find the Best Move**  
Analyze the game state, predict the entities' next moving turn and select your move from available actions in the current turn.

## **Given Game State:**
'''
STAY_COMPLETION = """Stay at (x = {x}, y = {y})"""