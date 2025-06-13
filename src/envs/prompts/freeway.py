LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer within \\boxed{}.'''

SLOW_AGENT_PROMPT = '''
Given a player starting at \((0, pos)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)), determine the minimal number of turns \(T\) and a sequence of actions \(\{a_t\}_{t=1}^T\) (\(a_t \in \{U, D, S\}\)) to reach \((0, 9)\), avoiding collisions with cars on freeways \(y = 1, \dots, 8\). 

**Constraints:**
1. **Player Movement:**  
   \(y_t = y_{t-1} + \Delta y_t\), where \(\Delta y_t = 
   \begin{cases} 
   +1 & \text{if } a_t = U, \\
   -1 & \text{if } a_t = D, \\
   0 & \text{if } a_t = S.
   \end{cases}\)  
   \(y_t \in [0, 9]\) for all \(t\).

2. **Car Dynamics:**  
   For each car on freeway \(k\) (\(t = 0\) data and probably a few data of \(t >= 1\) turns provided below), its span at turn \(t\) is:  
   - **Left-moving:** Span = \([h_k - s_k t,\ \tau_k - s_k t]\)  
   - **Right-moving:** Span = \([\tau_k + s_k t,\ h_k + s_k t]\)  
   Collision occurs if \(0 \in \text{Span}\) at \(y = y_t\) for any car on freeway \(y_t\).  

**Given Car Data (for \(y = 1, \dots, 8\)):**  
- **Freeway \(k\):** \(n_k\) cars, each with initial head \(h_{k,i}\), tail \(\tau_{k,i}\), direction \(d_{k,i}\), speed \(s_{k,i}\).  
  (Specific values provided in the original game state.)  

**Objective:**  
Find a sequence of actions \(\{a_t\}_{t=1}^T\) which minimizes \(T\) such that \(y_T = 9\) and \(0 \notin \text{Span}_{k,i}(t)\) for all \(t \leq T\) and cars on \(y = y_t\).  
'''


# Sample prompts for the fast agent, which uses a previous thinking model's output as a reference.
# ‼️ Template Para: Need to provide the current game state and a past thinking model's output.
FAST_AGENT_PROMPT = '''
You are a player in a freeway game, starting at \((0, y_0)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)). Your goal is to reach \((0, 9)\) while avoiding collisions with cars on freeways \(y = 1, \dots, 8\).

---

### 1. **Current State (Turn \(t\)):**  
- Player position: \( (0, y_t) \)  
- Observed car spans \(\{\text{Span}_{k,i}(t)\}\) for freeways \(k = 1, \dots, 8\)

<Current game state data provided here>

---

### 2. **Guidance from a Previous Thinking Model (Turn \(t_0 < t\)):**  
You have access to a past output from a thinking model, computed at turn \(t_0\) based on then-current observations. It includes a proposed **action sequence**. This guidance may no longer perfectly match the current situation but can still be valuable for decision-making.

> **Thinking Model Output (from Turn \(t_0\)):**
>
> #### Action Sequence (starts from \(t_0+1\)):
> Turn \(t_0+1\): D  
> Turn \(t_0+2\): S  
> Turn \(t_0+3\): U  
> Turn \(t_0+4\): U  
> ...  

Use this plan as a **strategic reference**, not a mandatory instruction. Consider how much of the original strategy is still valid under the current dynamics.

---

### 3. **Game Dynamics:**

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
  - Left-moving: \([h_{k,i} - s_{k,i}(t+1), \tau_{k,i} - s_{k,i}(t+1)]\)  
  - Right-moving: \([\tau_{k,i} + s_{k,i}(t+1), h_{k,i} + s_{k,i}(t+1)]\)

---

### 4. **Task (Turn \(t\)):**

Choose **one** action \(a_t \in \{U, D, S\}\) for the current turn, with the following considerations:

- **Collision Avoidance:**  
  Ensure the action avoids both immediate and near-future collisions.

- **Strategic Consistency (Optional):**  
  Reference the thinking model’s prior strategy. If the current environment still aligns with its assumptions, you may choose to continue along the same strategic direction. If not, adapt as needed.

**Answer Format**:
\\boxed{
a_t
}
'''

FAST_AGENT_CONCLUSION_PROMPT = '''
You are a player in a freeway game, starting at \((0, y_0)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)). Your goal is to reach \((0, 9)\) while avoiding collisions with cars on freeways \(y = 1, \dots, 8\).

---

### 1. **Current State (Turn \(t\)):**  
- Player position: \( (0, y_t) \)  
- Observed car spans \(\{\text{Span}_{k,i}(t)\}\) for freeways \(k = 1, \dots, 8\)

<Current game state data provided here>

---

### 2. **Guidance from a Previous Thinking Model (Turn \(t_0 < t\)):**  
You have access to a past output from a thinking model, computed at turn \(t_0\) based on then-current observations. It includes a proposed **action sequence** and a **main strategy explanation**. This guidance may no longer perfectly match the current situation but can still be valuable for decision-making.

> **Thinking Model Output (from Turn \(t_0\)):**
>
> #### Action Sequence (starts from \(t_0+1\)):
> Turn \(t_0+1\): D  
> Turn \(t_0+2\): S  
> Turn \(t_0+3\): U  
> Turn \(t_0+4\): U  
> ...  
>
> #### Main Strategy:
> The plan was to descend early to row 3 to avoid mid-row congestion at \(t=2\), then ascend through safe rows during \(t=3\) to \(t=9\), avoiding late-phase threats on rows 7-8 at \(t=10,11,12\).

Use this plan as a **strategic reference**, not a mandatory instruction. Consider how much of the original strategy is still valid under the current dynamics.

---

### 3. **Game Dynamics:**

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
  - Left-moving: \([h_{k,i} - s_{k,i}(t+1), \tau_{k,i} - s_{k,i}(t+1)]\)  
  - Right-moving: \([\tau_{k,i} + s_{k,i}(t+1), h_{k,i} + s_{k,i}(t+1)]\)

---

### 4. **Task (Turn \(t\)):**

Choose **one** action \(a_t \in \{U, D, S\}\) for the current turn, with the following considerations:

- **Collision Avoidance:**  
  Ensure the action avoids both immediate and near-future collisions.

- **Strategic Consistency (Optional):**  
  Reference the thinking model’s prior strategy. If the current environment still aligns with its assumptions, you may choose to continue along the same strategic direction. If not, adapt as needed.

**Answer Format**:
\\boxed{
a_t
}
'''

DEFAULT_ACTION = "U" 

ALL_ACTIONS = "UDS"

SEQUENCE_FORMAT_PROMPT = '''
**Answer Format**:

\\boxed{
Turn 1: a_1(i.e, action_1)
Turn 2: a_2
...
Turn t: a_t
}

Where each action \(a_t \in \{\text{U (up)},\ \text{D (down)},\ \text{S (stay)}\}\).
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
'''

