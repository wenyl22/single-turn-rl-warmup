LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer, a capital letter, within \boxed{}. (e.g. \boxed{A})'''

LLM_BASE_PROMPT = '''
# **Freeway Game: Optimal Action Selection**  

## **Game Overview**  
You are playing "Freeway", a game where you must guide your character safely across multiple lanes of moving traffic. Your goal is to **reach the destination (y = 9) from the starting point (y = 0) in the fewest turns while avoiding collisions with cars.**  

Imagine a **2D grid** where:  
- **The vertical axis (y)** represents different freeways, numbered from `0` to `9`.  
- **The horizontal axis (x)** represents positions along each freeway.  

You always stay at **x = 0**, meaning you cannot move left or right. Instead, your only movement options are:  
- Moving **up** (to a higher freeway, y → y + 1).  
- Moving **down** (to a lower freeway, y → y - 1).  
- Staying **on the same freeway** (no movement).  

## **Game Mechanics**  

### **1. Freeways & Cars**  
Each freeway (from y = 1 to y = 8) may have cars moving left or right.  
- **Cars move at a fixed speed per turn** (e.g., a speed of 3 means the car moves 3 units in the x-direction each turn).  
- **Each car has a span**, which is the range of x-values it occupies.  
- **Movement Direction**:  
  - If a car moves **right**, its span extends as it moves forward.  
  - If a car moves **left**, its span moves backward.  

**Example:**  
A car on Freeway 2 with a **head position at x = 18** and a **tail position at x = 29** moves **left at a speed of 6**.  
- After **one turn**, its new span will be **[head: 12, tail: 23]**.  
- After **two turns**, its span will be **[head: 6, tail: 17]**.  

### **2. Collisions**  
A **collision happens if after your move(or stay), your position (x = 0, y) overlaps with a car's span**.  
- If a collision occurs, you are **reset to the starting position (0, 0)**.  
- To avoid collisions, you must predict car movements and time your actions carefully.  

## **Game State Representation**  
Each turn, you receive the **current game state**, which includes:  
- Your position: (x = 0, y).  
- Car information for each freeway (y = 1 to 8). Each may have zero, one or multiple cars, each represented by:: 
  - Head position (front of the car).  
  - Tail position (back of the car).  
  - Direction (left or right).  
  - Speed (how many x-units the car moves per turn).

## **Your Task: Find the Best Move**  
Each turn, you must **analyze car positions, predict future movements, and decide the safest and most efficient action** from the following options:  
- **A**: Move **up** to Freeway (y + 1).  
- **B**: Move **down** to Freeway (y - 1).  
- **C**: Stay on the current freeway.  

## **Given Game State:**
'''
STAY_COMPLETION = f"""Stay in the same freeway"""