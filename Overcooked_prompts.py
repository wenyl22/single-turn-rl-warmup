LLM_SYSTEM_PROMPT = "You are a friendly chat assistant who is correct and brief at all times."

RULES = f'''Players must coordinate to make onion soups with 3 onions each. Once a soup is cooked it needs to be placed on a plate and delivered. Players can only carry one item at a time. A soup can only be loaded onto plate by a player if they are holding a plate. The goal is to maximize the number of deliveries.'''

BASE_PROMPT="""I am {player_name}. I am playing the game Overcooked with my partner {other_player_name}. {envDescription} """ + f'''Overcooked has the following rules: {RULES}. We have agreed to be efficient and prepare for the next soup while the current soup is cooking. I'll provide my current state, teammate's status, and my possible actions. Help me select the best action from the list. Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format'''
