LLM_SYSTEM_PROMPT = '''Please think step by step. Then put your final answer within \\boxed{}.'''
SLOW_AGENT_PROMPT = '''
Now a player is playing a multi-turn game, and suppose current turn is \{t_1\}. You will be given a plate starting at \((x_{t_1}, 0)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)) and some rewards dropping from the ceiling. Determine the way the plate moves in a sequence of actions \(\{a_{t_1 + t\}_{t=0}^{H - 1}\) (\(a_{t} \in \{L, R, S\}\)) to collect the maximum total reward.
---
### Core Rules
**1. Player Movement:**  
   \(x_{t+1} = x_t + \Delta x_t\), where \(\Delta x_t = 
   \begin{cases} 
   -1 & \text{if } a_t = L, \\
   +1 & \text{if } a_t = R, \\
   0 & \text{if } a_t = S.
   \end{cases}\)  
   \(x_{t+1} \in [0, 9]\) for all \(t\).

**2. Reward Dynamics:**
   Suppose some reward is of value r and moves downwards from the ceiling. Its position \((x, y)\) and speed is \(s\) at turn \(t_1\). Then at turn \(T > t_1\), the reward's position becomes:
    - \(x_i(T) = x\), \(y_i(T) = y - s(T - t_1)\).
    - The reward  is collected at turn T, if the plate's position \(x_T = x(T)\) and \(y(T - 1) > 0, y(T) \leq 0\).
---
### Task:
Choose a sequence of actions \(\{a_{t_1 + t}\}_{t=0}^{H - 1}\) to maximize the sum of collected reward by the plate in the next few turns, where \(H\) is the number of turns you plan to play.
'''

ACTION_FORMAT_PROMPT = '''
**Answer Format**:

\\boxed{
Turn t_1: a_\{t_1\}
Turn t_1 + 1: a_\{t_1 + 1\}
...
}

Where each action \(a_t \in \{L, R, S\}\).
---
### **Current State (Turn \(t_1\)):**
'''

CONCLUSION_FORMAT_PROMPT = '''
**Answer Format**:

Your answer **must** include both of the following, clearly separated:

**(1) Action Sequence (in order):**

\\boxed{
Turn t_1: a_\{t_1\}
Turn t_1 + 1: a_\{t_1 + 1\}
...
}

Where each action \(a_t \in \{L, R, S\}\).

**(2) Main Thinking Conclusion (one or two sentences):**

A concise summary explaining the main decision strategy behind your chosen sequence. 
---
### **Current State (Turn \(t_1\)):**
'''


FAST_AGENT_PROMPT = '''
Now a player is playing a multi-turn game, and suppose current turn is \{t_0\}. You will be given a plate starting at \((x_{t_0}, 0)\) on a 2D grid (vertical axis \(y = 0, 1, \dots, 9\)) and some rewards dropping from the ceiling. 
---
### Core Rules
**1. Player Movement:**  
   \(x_{t+1} = x_t + \Delta x_t\), where \(\Delta x_t = 
   \begin{cases} 
   -1 & \text{if } a_t = L, \\
   +1 & \text{if } a_t = R, \\
   0 & \text{if } a_t = S.
   \end{cases}\)  
   \(x_{t+1} \in [0, 9]\) for all \(t\).

**2. Reward Dynamics:**
   Suppose some reward is of value r and moves downwards from the ceiling. Its position \((x, y)\) and speed is \(s\) at turn \(t_0\). Then at turn \(T > t_0\), the reward's position becomes:
    - \(x_i(T) = x\), \(y_i(T) = y - s(T - t_0)\).
    - The reward  is collected at turn T, if the plate's position \(x_T = x(T)\) and \(y(T - 1) > 0, y(T) \leq 0\).

**3. Guidance from a Previous Thinking Model (Turn \(t_1 \leq t_0\)):**  
    Sometimes, you have access to a past output from a thinking model, computed at turn \(t_1\) based on then-current observations. This guidance may no longer perfectly match the current situation but can still be valuable for decision-making. You can use this plan as a **strategic reference**, not a mandatory instruction. Consider how much of the original strategy is still valid under the current dynamics.
---
### Task
Choose **one** action \( a_{t_0} \in \{L, R, S\} \) for turn \( t_0 \), adhering to:
1. **Reward Collection:**  
   Ensure the action in this turn maximizes the total reward collected by the plate in the next few turns.
2. **Strategic Consistency (Optional):**  
  Refer to the thinking model's prior strategy. If the current environment still aligns with its assumptions, you may choose to continue along the same strategic direction. If not, adapt as needed.
**Answer Format**:
\\boxed{
a_{t_0}
}
---
### **Current State (Turn \(t_0\)):**
'''

DEFAULT_ACTION = 'S'

ALL_ACTIONS = 'LRS'