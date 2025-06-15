LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer within \\boxed{}.'''

SLOW_AGENT_PROMPT = '''
Now a player is playing a multi-turn game, and suppose current turn is \{t_0\}. Given the inital position \((0, y_{t_0})\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)), determine the minimal number of turns \(T\) and a sequence of actions \(\{a_{t_0 + t}\}_{t=1}^T\) to reach \((0, 9)\), avoiding collisions with cars on freeways \(y = 1, \dots, 8\). 
---
### 1. **Game Dynamics:**
- Player update:  
  \( y_{t+1} = y_t + \Delta y_{t+1} \), where  
  \[
  \Delta y_{t+1} = \begin{cases} 
  +1 & \text{if } a_{t+1} = U \\ 
  -1 & \text{if } a_{t+1} = D \\ 
  0 & \text{if } a_{t+1} = S 
  \end{cases}, \quad y_{t+1} \in [0,9]
  \]
- Car update rules: 
    For car \(k\) on freeway \(i\), suppose its head is at \(h_{k, i}\), tail is at \(\tau_{k, i}\) at turn \(t_0\), and speed is \(s_{k, i}\). Then at turn \(T > t_0\), the car span becomes:
  - Left-moving: \([h_{k, i} - s_{k, i} (T-t_0), \tau_{k, i} - s_{k, i} (T-t_0)]\)
  - Right-moving: \([\tau_{k, i} + s_{k, i} (T-t_0), h_{k, i} + s_{k, i} (T-t_0)]\)
    Collision occurs if \( 0 \in \text{Span}_{k,i}(t) \) at \(y = y_t\) for any car on freeway \(y_t\).
---
### 2. **Task (Turn \(t_0\)):**
Find a sequence of actions \(\{a_{t_0 + t\}_{t=1}^T\) which minimizes \(T\) such that \(y_{t_0 + T} = 9\) and \(0 \notin \text{Span}_{k,i}(t)\) for all \(t \leq T\) and cars on \(y = y_t\).  
'''

ACTION_FORMAT_PROMPT = '''
**Answer Format**:

\\boxed{
Turn 1: a_1(i.e, action_1)
Turn 2: a_2
...
Turn t: a_t
}

Where each action \(a_t \in \{\text{U (up)},\ \text{D (down)},\ \text{S (stay)}\}\).
---
### 3. **Current State (Turn \(t_0\)):**
'''

CONCLUSION_FORMAT_PROMPT = '''
**Answer Format**:

Your answer **must** include both of the following, clearly separated:

**(1) Action Sequence (in order):**

\\boxed{
Turn 1: a_1(i.e, action_1)
Turn 2: a_2
...
Turn t: a_t
}

Where each action \(a_t \in \{\text{U (up)},\ \text{D (down)},\ \text{S (stay)}\}\).

**(2) Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 
---
### 3. **Current State (Turn \(t_0\)):**
'''

FAST_AGENT_ACTION_PROMPT = '''
You are a player in a freeway game, starting at \((0, y_\{t_0\})\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)). Your goal is to reach \((0, 9)\) while avoiding collisions with cars on freeways \(y = 1, \dots, 8\).
---
### 1. **Game Dynamics:**
- Player update:  
  \( y_{t+1} = y_t + \Delta y_{t+1} \), where  
  \[
  \Delta y_{t+1} = \begin{cases} 
  +1 & \text{if } a_{t+1} = U \\ 
  -1 & \text{if } a_{t+1} = D \\ 
  0 & \text{if } a_{t+1} = S 
  \end{cases}, \quad y_{t+1} \in [0,9]
  \]
- Car update rules: 
    For car \(k\) on freeway \(i\), suppose its head is at \(h_{k, i}\), tail is at \(\tau_{k, i}\) at turn \(t_0\), and speed is \(s_{k, i}\). Then at turn \(T > t_0\), the car span becomes:
  - Left-moving: \([h_{k, i} - s_{k, i} (T-t_0), \tau_{k, i} - s_{k, i} (T-t_0)]\)
  - Right-moving: \([\tau_{k, i} + s_{k, i} (T-t_0), h_{k, i} + s_{k, i} (T-t_0)]\)
    Collision occurs if \( 0 \in \text{Span}_{k,i}(t) \) at \(y = y_t\) for any car on freeway \(y_t\).
---
### 2. **Guidance from a Previous Thinking Model (Turn \(t_1 \leq t_0\)):**  
Sometimes, you have access to a past output from a thinking model, computed at turn \(t_1\) based on then-current observations. It includes a proposed **action sequence**. This guidance may no longer perfectly match the current situation but can still be valuable for decision-making. You can use this plan as a **strategic reference**, not a mandatory instruction. Consider how much of the original strategy is still valid under the current dynamics.
---
### 3. **Task (Turn \(t_0\)):**
Choose **one** action \(a_{t_0 + 1} \in \{U, D, S\}\) for the current turn, with the following considerations:
- **Collision Avoidance:**  
  Ensure the action avoids both immediate and near-future collisions.
- **Strategic Consistency (Optional):**  
  Reference the thinking model's prior strategy. If the current environment still aligns with its assumptions, you may choose to continue along the same strategic direction. If not, adapt as needed.
**Answer Format**:
\\boxed{
a_{t_0 + 1}
}
---
### 4. **Current State (Turn \(t_0\)):**  
'''

FAST_AGENT_CONCLUSION_PROMPT = '''
You are a player in a freeway game, starting at \((0, y_\{t_0\})\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)). Your goal is to reach \((0, 9)\) while avoiding collisions with cars on freeways \(y = 1, \dots, 8\).
---
### 1. **Game Dynamics:**
- Player update:  
  \( y_{t+1} = y_t + \Delta y_{t+1} \), where  
  \[
  \Delta y_{t+1} = \begin{cases} 
  +1 & \text{if } a_{t+1} = U \\ 
  -1 & \text{if } a_{t+1} = D \\ 
  0 & \text{if } a_{t+1} = S 
  \end{cases}, \quad y_{t+1} \in [0,9]
  \]
- Car update rules: 
    For car \(k\) on freeway \(i\), suppose its head is at \(h_{k, i}\), tail is at \(\tau_{k, i}\) at turn \(t_0\), and speed is \(s_{k, i}\). Then at turn \(T > t_0\), the car span becomes:
  - Left-moving: \([h_{k, i} - s_{k, i} (T-t_0), \tau_{k, i} - s_{k, i} (T-t_0)]\)
  - Right-moving: \([\tau_{k, i} + s_{k, i} (T-t_0), h_{k, i} + s_{k, i} (T-t_0)]\)
    Collision occurs if \( 0 \in \text{Span}_{k,i}(t) \) at \(y = y_t\) for any car on freeway \(y_t\).
---
### 2. **Guidance from a Previous Thinking Model (Turn \(t_1 \leq t_0\)):**  
Sometimes, you have access to a past output from a thinking model, computed at turn \(t_1\) based on then-current observations. It includes a proposed **action sequence** and a **main strategy explanation**. This guidance may no longer perfectly match the current situation but can still be valuable for decision-making. You can use this plan as a **strategic reference**, not a mandatory instruction. Consider how much of the original strategy is still valid under the current dynamics.
---
### 3. **Task (Turn \(t_0\)):**
Choose **one** action \(a_{t_0 + 1} \in \{U, D, S\}\) for the current turn, with the following considerations:
- **Collision Avoidance:**  
  Ensure the action avoids both immediate and near-future collisions.
- **Strategic Consistency (Optional):**  
  Reference the thinking model's prior strategy. If the current environment still aligns with its assumptions, you may choose to continue along the same strategic direction. If not, adapt as needed.
**Answer Format**:
\\boxed{
a_{t_0 + 1}
}
---
### 4. **Current State (Turn \(t_0\)):**  
'''

DEFAULT_ACTION = "U" 

ALL_ACTIONS = "UDS"