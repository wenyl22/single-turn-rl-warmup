Help Alice collaborate with Bob in *Overcooked* to maximize the total reward from delivered soup recipes. Optimize coordination, planning, and decision-making based on the current game state and action history of both players. 
---

### **Game Rules & Mechanics**  
#### **1. Game Grid (2D Tile Types)**  
- **Empty Tile:** Walkable space.  
- **Dispensers:** Sources for items (tomato/onion/clean plate).  
- **Pot:** Cooks soup (3 ingredients → auto-starts cooking).  
- **Serving Counter:** Deliver cooked soup for rewards.  
- **Kitchen Counter:** Temporary item storage.  

#### **2. Player Actions per Turn**  
Each player (Alice/Bob) can:  
- **Move** (L/R/U/D): Changes position *and* orientation. Specifically, L (x-1, y), R (x+1, y), U (x, y+1), D (x, y-1).
- **Change Orientation** (L/R/U/D): No move if the adjacent tile is not an empty one.  
- **Interact (I):** Player can interact with a non-empty tile if it is 1) standing on an empty tile adjacent to the tile and 2) facing the tile. Interact effects depend on game state and tile type:
  - **Dispensers:** Grab item (if player is empty-handed).
  - **Pot:**  
    - Add ingredient (if pot is not full and player is holding ingredient) → cooks automatically at 3 ingredients.  
    - Serve soup (if player is holding a clean plate and pot has cooked soup).
  - **Serving Counter:** Deliver soup (if player is holding a soup in plate).
  - **Kitchen Counter:** Pick up items (if player is empty-handed and counter is occupied) or place items (if player is holding an item and counter is empty).
- **Stay (S):** Do nothing.

#### **3. Key Constraints**  
- Players can carry **one item at a time**.  
- Players cannot walk onto/through each other.
- Players cannot walk onto/through non-empty tiles. 

#### **4. Recipes & Rewards**
- Each recipe requires different ingredients, cooking time, and yields varying rewards.
- Cooking time is 20 turns if ingredients do not match any recipe in the order, and serving the recipe does not yield rewards.

### **Your Task**  

Your goal is to maximize recipe rewards by generating an executable python function `next_action(json_state)` with the input `json_state` representing the current game state as a JSON object. The function should analyze the game state and return the next action for Alice, represented as a single character string:
- 'U' for Up
- 'D' for Down
- 'L' for Left
- 'R' for Right
- 'I' for Interact
- 'S' for Stay
Notice that the code will be executed in a loop, so it should return the next action each time it is called with the current game state, which will **change after each action**.

### **Input Format**
The input `json_state` is a JSON object containing:
- **environment:** Details about the game grid, tile types, and recipe requirements.
- **game_state:** Current turn, player positions, orientations, inventories, action histories, and object states (e.g., pot contents).
You should analyze the history of actions taken by each player, inferring their goals and strategies to determine the best strategy for Alice.

### **Output Requirements**
Generate **two clearly labeled parts**:

**Part 1: Summary**  
\boxed{One-sentence intent describing your strategy for the next actions }  

**Part 2: Python Function**  
```python
def next_action(json_state) -> str:
    """Returns one of U/D/L/R/I/S based on game state analysis each time this function is called.
    
    Args:
        json_state: The current game state as JSON object
        
    Returns:
        str: Single character representing the next action 
        ('U', 'D', 'L', 'R', 'I', 'S')
    """
    # Your logic here
    return action
```

#### **Example Output Format**

**Part 1: Summary**
\boxed{Move toward onion dispenser to gather missing ingredient for the pot}

**Part 2: Python Function**
```python
def next_action(json_state: dict) -> str:
    # Implementation...
    if json_state['game_state']['players']['Alice']['position'] == [2, 1]:
        return 'I'  # Interact with the onion dispenser
    else :
        # Move towards the onion dispenser
        return 'R' # Example action, more logic needed based on game state
```


### `json_state`

```python
json_state = {
    "environment": {
        "tile_types": {
            "Kitchen Counter": [[0, 0], [1, 0], [2, 0], [5, 0], [6, 0], [7, 0], [0, 1], [7, 1], [2, 2], [3, 2], [4, 2], [5, 2], [0, 3], [7, 3], [0, 4], [1, 4], [2, 4], [5, 4], [6, 4], [7, 4]],
            "Tomato Dispenser": [],
            "Onion Dispenser": [[3, 4], [4, 4]],
            "Plate Dispenser": [[0, 2]],
            "Pot": [[3, 0], [4, 0]],
            "Serving Counter": [[7, 2]]
        },
        "recipe": {
            "onions": 3,
            "tomatoes": 0,
            "reward": 20,
            "cook_time": 5
        }
    },
    "game_state": {
        "turn": 29,
        "players": {
            "Alice": {
                "position": [2, 1],
                "orientation": "U",
                "holding": None,
                "action_history": ["L", "D", "I", "L", "U"]
            },
            "Bob": {
                "position": [4, 3],
                "orientation": "R",
                "holding": ["onion", 1],
                "action_history": ["I", "R", "U", "I", "R"]
            }
        },
        "objects": {
            "Kitchen Counter (2, 2)": ["onion", 1],
            "Pot (3, 0)": {
                "onions": 2,
                "tomatoes": 0,
                "cooking": False,
            }
        }
    }
}
```