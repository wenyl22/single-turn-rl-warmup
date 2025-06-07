LLM_SYSTEM_PROMPT = """You are an AI agent playing the cooperative cooking game Overcooked. Your goal is to prepare and deliver as many dishes as possible within the time limit. You must coordinate with your teammate efficiently, plan ahead, and make optimal decisions based on the current game state. Please analyze the game state carefully and select the best action from the available options. Put your final answer within \\boxed{}."""

LLM_BASE_PROMPT="""

# Overcooked LLM Agent 

## Game Rules and Mechanics
- Players can only carry one item at a time
- The player needs to place 3 onions to in a pot, then a cook will be automatically initiated
- Cooking time for soup is 20 time steps
- After a soup is cooked, the player must use a clean plate to serve and deliver it
- Delivery is only possible at designated delivery points
- Players can place and pick up items from counters
- Players cannot walk through each other or obstacles

## Environment Details
{env_description}
"""

GAME_STATE_PROMPT="""
## Current Game State

### Player Information
- **You ({my_name})**
  - Holding: {my_holding}

- **Teammate ({he_name})**
  - Holding: {he_holding}

### Object Position and States
{object_states}

## Available Actions
{available_actions}
"""

STAY_COMPLETION = "wait."

MAPPING ={
    'c': 'Pot',
    'o': 'Onion Dispenser',
    'p': 'Plate Dispenser',
    'd': 'Delivery Point',
    's': 'Storage Counter',
    'k': 'Kitchen Counter',
    'g': 'Gate',
}