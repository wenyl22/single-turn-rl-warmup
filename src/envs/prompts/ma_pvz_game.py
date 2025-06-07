LLM_SYSTEM_PROMPT = """Please think step by step and put your final answer within \\boxed{}."""

GAME_PROMPT = """
You are required to participate in a simplified version of the Plants vs. Zombies game. The game is played on a 5×7 board, where zombies spawn from the far right side and move one step to the left each turn. The types of plants and zombies are as follows:

## Plants  
- Sunflower (X): Costs 50 sun, has 2 HP, and generates an extra 10 sun each turn.  
- Peashooter (W): Costs 100 sun, has 2 HP, and deals 1 damage each turn to the first zombie in its current row.  
- Three-Line Shooter (S): Costs 325 sun, has 2 HP, and deals 1 damage each turn to the first zombie in its current row as well as the first zombie in each of the adjacent rows.  
- Wall-nut (J): Costs 50 sun and has 10 HP.  
- Torch Stump (H): Costs 125 sun, has 2 HP; it increases the damage of the plant to its left in the same row (applied directly to the plant rather than to a projectile) by +1, and this effect can only be applied once.  
- Fire Chili (F): Costs 300 sun and eliminates all zombies in its row.

## Zombies  
- Regular Zombie (N): Has 4 HP and deals 1 damage each turn to the plant that blocks its path.  
- Roadblock Zombie (R): Has 8 HP and deals 1 damage each turn to the plant that blocks its path.  
- Bucket Zombie (B): Has 12 HP and deals 1 damage each turn to the plant that blocks its path.  
- High-Attack Zombie (I): Has 6 HP and deals 3 damage each turn to the plant that blocks its path.

## Rules  
- The board is a 5x7 grid, where columns are numbered from 0 to 6 (left to right) and rows are numbered from 0 to 4 (top to bottom).
- At least 25 sun is gained each turn.
- A new zombie is spawned every 5 turns.
- After every 10 turns, newly spawned zombies have their HP increased by 4, and the number of zombies spawned increases by 1.
- Your score increases by 1 each turn.
- The game lasts for a maximum of 100 turns.
- Plants cannot be placed on the same grid cell, but zombies can coexist in the same cell. 0 means empty cell.
- There are no lawn mowers.
- Roadblock Zombies only spawn after turn 10, and Bucket Zombies and High-Attack Zombies only spawn after turn 20.

## Answer Format

Please output the plants to be planted in the next several turns. Output in the format "PlantType Row Column"(e.g. X 2 1). If multiple plants need to be planted, separate them using a semicolon (`;`).  If you do not want to plant any plants, output "None" in that turn.

\\boxed{
    Turn t: PlantType Row Column; PlantType Row Column; ...
    Turn t + 1: 
    ...
}
"""


GAME_PROMPT_LOW_LEVEL = """
You are required to participate in a simplified version of the Plants vs. Zombies game. The game is played on a 5×7 board, where zombies spawn from the far right side and move one step to the left each turn. The types of plants and zombies are as follows:

## Plants  
- Sunflower (X): Costs 50 sun, has 2 HP, and generates an extra 10 sun each turn.  
- Peashooter (W): Costs 100 sun, has 2 HP, and deals 1 damage each turn to the first zombie in its current row.  
- Three-Line Shooter (S): Costs 325 sun, has 2 HP, and deals 1 damage each turn to the first zombie in its current row as well as the first zombie in each of the adjacent rows.  
- Wall-nut (J): Costs 50 sun and has 10 HP.  
- Torch Stump (H): Costs 125 sun, has 2 HP; it increases the damage of the plant to its left in the same row (applied directly to the plant rather than to a projectile) by +1, and this effect can only be applied once.  
- Fire Chili (F): Costs 300 sun and eliminates all zombies in its row.

## Zombies  
- Regular Zombie (N): Has 4 HP and deals 1 damage each turn to the plant that blocks its path.  
- Roadblock Zombie (R): Has 8 HP and deals 1 damage each turn to the plant that blocks its path.  
- Bucket Zombie (B): Has 12 HP and deals 1 damage each turn to the plant that blocks its path.  
- High-Attack Zombie (I): Has 6 HP and deals 3 damage each turn to the plant that blocks its path.

## Rules  
- The board is a 5x7 grid, where columns are numbered from 0 to 6 (left to right) and rows are numbered from 0 to 4 (top to bottom).
- At least 25 sun is gained each turn.
- A new zombie is spawned every 5 turns.
- After every 10 turns, newly spawned zombies have their HP increased by 4, and the number of zombies spawned increases by 1.
- Your score increases by 1 each turn.
- The game lasts for a maximum of 100 turns.
- Plants cannot be placed on the same grid cell, but zombies can coexist in the same cell.0 means empty cell.
- There are no lawn mowers.
- Roadblock Zombies only spawn after turn 10, and Bucket Zombies and High-Attack Zombies only spawn after turn 20.

## Task

Please determine the plants to be planted in the next one turn. Output in the format "PlantType Row Column"(e.g. X 2 1). If multiple plants need to be planted, separate them using a semicolon (`;`).  If you do not want to plant any plants, output "None" in that turn.

Besides the current state as map, you probably also receive a plan advice(or not), which is a sequence of advised actions in the next few turns. Each turn's advice is in the format "Turn x: PlantType Row Column; PlantType Row Column; ...". This is only a reference which may not be optimal. You can choose to follow the advice and output its actions for the next one turn, or choose your own actions.

## Answer Format

\\boxed{PlantType Row Column; PlantType Row Column; .../ None}

"""
