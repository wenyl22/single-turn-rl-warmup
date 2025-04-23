Supervisor Agent

# **Game Overview**  

**Freeway** is a game where the player must guide the character safely across multiple lanes of moving traffic. The goal is to **reach the destination (y = 9) from the starting point (y = 0) in fewest turns while avoiding collisions with cars.**  

Imagine a **2D grid** where:  
- **The vertical axis (y)** represents different freeways, numbered from `0` to `9`.  
- **The horizontal axis (x)** represents positions along each freeway.  

Player always stay at **x = 0**, meaning it cannot move left or right. Instead, its only movement options are:  
- Moving **up** (to a higher freeway, y → y + 1).  
- Moving **down** (to a lower freeway, y → y - 1).  
- Staying **on the same freeway** (no movement).  

# **Game Mechanics**  

## **1. Freeways & Cars**  
Each freeway (y = 1 to 8) may have at most one moving car.  
- **Cars move at a fixed speed per turn** (e.g., a speed of 3 means the car moves 3 units in the x-direction each turn).  
- **Each car has a span**, which is the range of x-values it occupies.  
- **Movement Direction**:  
  - If a car moves **right**, its span extends as it moves forward.  
  - If a car moves **left**, its span moves backward.  

**Example:**  
A car on Freeway 2 with a **head position at x = 18** and a **tail position at x = 29** moves **left at a speed of 6**.  
- After **one turn**, its new span will be **[head: 12, tail: 23]**.  
- After **two turns**, its span will be **[head: 6, tail: 17]**.  

## **2. Collisions**  
A **collision happens if, at any point before or after the player's move, its position (x = 0, y) overlaps with a car's span**.  
- If a collision occurs, the player is **reset to the starting position (0, 0)**.  
- To avoid collisions, the player must predict car movements and time its actions carefully.  

## **Game State Representation**  
Each turn, the player receive the **current game state**, which includes:  
- Player position: (x = 0, y).  
- Plan Scratch Pad: [Previous plan or empty].
- Cars information: on each freeway (y = 1 to 8)**:  
  - **Head position** (front of the car).  
  - **Tail position** (back of the car).  
  - **Direction** (left or right).  
  - **Speed** (how many x-units the car moves per turn).
- Available actions: A subset of moving up, down, or staying.

```md
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
\boxed\{[Plan Agent | Follow Plan Agent | React Agent]\}  
```

## **Current Game State**:  
- Player Position: (0, 0).
- Plan Scratch Pad: Empty.
- Cars on each freeway:
        - Freeway 1: head at **x = 8**, tail at **x = -3**, direction = right, speed = 4.
        - Freeway 2: head at **x = 36**, tail at **x = 25**, direction = right, speed = 6.
        - Freeway 3: head at **x = 36**, tail at **x = 25**, direction = right, speed = 6.
        - Freeway 4: head at **x = 0**, tail at **x = -11**, direction = right, speed = 12.
        - Freeway 5: head at **x = 24**, tail at **x = 35**, direction = left, speed = 12.
        - Freeway 6: head at **x = 36**, tail at **x = 25**, direction = right, speed = 6.
        - Freeway 7: head at **x = -6**, tail at **x = -17**, direction = right, speed = 3.
        - Freeway 8: head at **x = -6**, tail at **x = -17**, direction = right, speed = 3.
- Available actions:
A. Move up to Freeway 1
B. Stay in the same freeway



Plan Agent
```md
## **Your Task**:
Analyze the current game state and create or update a strategic plan for crossing the road safely. Plan also takes time, so be efficient and correct.
**Instructions**:
- 1. Predict car movements and plan the safest route to the destination.
- 2. Overwrite the plan scratch pad with your summarized reasoning and plan

## **Answer Format**:
\\boxed\{[new scratch pad content]\}
```

Follow Plan Agent
```md
## **Your Task**: 
Read the plan on the scratch pad and act accordingly. You are short in time, so do not overthink or plan for the future.

## **Answer Format**:  
\\boxed\{[MOVE UP/DOWN/STAY]\}  
```

React Agent
```md
## **Your Task**: 
Analyze the current game state and take immediate action to avoid a collision. You are short in time, so do not overthink or plan for the future.

## **Answer Format**:
\\boxed\{[MOVE UP/DOWN/STAY]\}  
```