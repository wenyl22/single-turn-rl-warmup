LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer within \\boxed{}.'''

MATH_PROMPT = '''
Given a player starting at \((0, pos)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)), you need to reach \((0, 9)\) with a sequence of actions \(\{a_t\}_{t=1}^T\) (\(a_t \in \{U, D, S\}\)). Determine \(a_1\) which avoids collisions with cars on freeways \(y = 1, \dots, 8\), and minimizes number of turns \(T\).

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
   For each car on freeway \(k\) (initial data provided below), its span at turn \(t\) is:  
   - **Left-moving:** Span = \([h_k - s_k t,\ \tau_k - s_k t]\)  
   - **Right-moving:** Span = \([\tau_k + s_k t,\ h_k + s_k t]\)  
   Collision occurs if \(0 \in \text{Span}\) at \(y = y_t\) for any car on freeway \(y_t\).  

**Given Car Data (for \(y = 1, \dots, 8\)):**  
- **Freeway \(k\):** \(n_k\) cars, each with initial head \(h_{k,i}\), tail \(\tau_{k,i}\), direction \(d_{k,i}\), speed \(s_{k,i}\).  
  (Specific values provided in the original game state.)  

**Objective:**  
Find \(a_1\) which minimizes \(T\) such that \(y_T = 9\) and \(0 \notin \text{Span}_{k,i}(t)\) for all \(t \leq T\) and cars on \(y = y_t\).  
'''
ANSWER_FORMAT = '''
**Answer Format**:
\\boxed{
a_1
}
---
'''

STAY_COMPLETION = '''Stay in the same freeway.'''

CAR_STATE = '''
| Freeway \( k \) | Cars (head \( h \), tail \( \tau \), direction \( d \), speed \( s \)) |  
|-----------------|------------------------------------------------------------------------|
'''

