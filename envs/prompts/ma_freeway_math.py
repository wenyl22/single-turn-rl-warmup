LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer within \\boxed{}.'''

MATH_PROMPT = '''
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

**Answer Format**:
\\boxed{
Turn 1: a_1
Turn 2: a_2
...
}
'''


MATH_PROMPT_LOW_LEVEL = '''
You are a player on a freeway game, starting at \((0, y_0)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)). Your goal is to reach \((0, 9)\) while avoiding collisions with cars on freeways \(y = 1, \dots, 8\).
1. **Current State:**  
   - Player position at turn \( 0 \): \( (0, y_0) \)  
   - Plan Advice(may exist or not given): Sequence of advised actions \(\{a_{i}^\text{adv}\}_{i=1}^{H}\) (horizon \( H \)), where \( a_{i}^\text{adv} \in \{U, D, S\} \). This is only a reference which may be neither safe nor optimal.
   - Observed car spans \(\{\text{Span}_{k,i}(t = 0)\}\) for freeways \( k = 1, \dots, 8 \).  

2. **Dynamics:**  
   - Player: \( y_{t+1} = y_t + \Delta y_{t+1} \), where \( \Delta y_{t+1} = 
     \begin{cases} 
     +1 & \text{if } a_{t+1} = U, \\ 
     -1 & \text{if } a_{t+1} = D, \\ 
     0 & \text{if } a_{t+1} = S. 
     \end{cases} \)  
     Constraint: \( y_{t+1} \in [0, 9] \).  
   - Cars: For each car on freeway \( k \), span at \( t+1 \) is:  
     - Left-moving: \([h_{k,i} - s_{k,i}(t+1), \tau_{k,i} - s_{k,i}(t+1)]\)  
     - Right-moving: \([\tau_{k,i} + s_{k,i}(t+1), h_{k,i} + s_{k,i}(t+1)]\).  

**Task:**  
Choose **one** action \( a_{1} \in \{U, D, S\} \) for turn \( 1 \), adhering to:  
1. **Collision Avoidance:**  
    Ensure the action in this turn does not lead to a collision with any car on freeway \( k \) at \( y = y_t \) for all cars \( i \) in the next few turns. Sometimes under the problem constraints, a wrong a_1 can lead to a must-collision **several turns later**. 
2. **Plan Adherence(Optional):**  
   The advice may have mistakes, you can take it as a reference. If you follow the advice, the action should be \( a_{1} = a_{1}^\text{adv} \).

**Answer Format**:
\\boxed{
a_1
}
---
'''  

CAR_STATE = '''
| Freeway \( k \) | Cars (head \( h \), tail \( \tau \), direction \( d \), speed \( s \)) |  
|-----------------|------------------------------------------------------------------------|
'''
