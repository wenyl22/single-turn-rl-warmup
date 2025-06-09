LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer within \\boxed{}.'''
SLOW_AGENT_PROMPT = '''
Given a plate starting at \((pos, 0)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)) and some rewards dropping from the ceiling. Determine the way the plate moves in a sequence of actions \(\{a_t\}_{t=1}^T\) (\(a_t \in \{L, R, S\}\)) to collect the maximum total reward.

**Constraints:**
1. **Player Movement:**  
   \(x_t = x_{t-1} + \Delta x_t\), where \(\Delta x_t = 
   \begin{cases} 
   -1 & \text{if } a_t = L, \\
   +1 & \text{if } a_t = R, \\
   0 & \text{if } a_t = S.
   \end{cases}\)  
   \(x_t \in [0, 9]\) for all \(t\).

2. **Reward Dynamics:**  
   Each reward \(r_i\) moves downwards from its initial position \((x_i, y_i)\) at speed \(s_i\). Given the initial positions and speeds, the position of each reward at turn \(t\) is:
    - \(x_{i, t} = x_i, y_{i,t} = y_i - s_i \cdot t\)
    The reward is collected, if the plate's position \(x_t = x_{i}\) when \(y_{i, t - 1} > 0\) and \(y_{i, t} <= 0\).

**Objective:**  
Maximize the total reward collected by the plate.

## **Answer Format**:
\\boxed{
Turn 1: a_1
Turn 2: a_2
...
}
'''

FAST_AGENT_PROMPT = '''
You are a plate on a reward collection game, starting at \((x_0, 0)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)). Your goal is to maximize collected rewards falling from the ceiling.

1. **Current State:**
    - Plate position at turn \( 0 \): \( (x_0, 0) \)  
    - Plan Advice(may exist or not given): Sequence of advised actions \(\{a_{i}^\text{adv}\}_{i=1}^{H}\) (horizon \( H \)), where \( a_{i}^\text{adv} \in \{L, R, S\} \). This is only a reference which may be neither safe nor optimal.
    - Observed rewards with their initial positions and speeds.

2. **Dynamics:**
    - Plate: \( x_{t+1} = x_t + \Delta x_{t+1} \), where \( \Delta x_{t+1} = 
      \begin{cases} 
      -1 & \text{if } a_{t+1} = L, \\ 
      +1 & \text{if } a_{t+1} = R, \\ 
      0 & \text{if } a_{t+1} = S. 
      \end{cases} \)  
      Constraint: \( x_{t+1} \in [0, 9] \).  
    - Rewards: For each reward \( i \), its position at \( t+1 \) is:  
      - \(x_{i, t+1} = x_i\), \(y_{i,t+1} = y_i - s_i(t+1)\).
      - The reward is collected, if the plate's position \(x_t = x_{i}\) when \(y_{i, t - 1} > 0\) and \(y_{i, t} <= 0\).

**Task:**
Choose **one** action \( a_{1} \in \{L, R, S\} \) for turn \( 1 \), adhering to:
1. **Reward Collection:**  
   Ensure the action in this turn maximizes the total reward collected by the plate in the next few turns.
2. **Plan Adherence(Optional):**
    The advice may have mistakes, you can take it as a reference. If you follow the advice, the action should be \( a_{1} = a_{1}^\text{adv} \).
**Answer Format**:
\\boxed{
a_1
}
'''
DEFAULT_ACTION = 'S'

ALL_ACTIONS = 'LRS'