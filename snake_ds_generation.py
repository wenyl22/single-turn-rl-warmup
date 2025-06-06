from envs.minatar.environment import Environment
from envs.snake import llm_state_builder
import pandas as pd
import queue
from copy import deepcopy
dxy_mapping = {1 : (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)} # LRDU
dir_mapping = {1: 'L', 2: 'R', 3: 'D', 4: 'U'}
def check_valid_move(state_for_llm, action):
    head_x, head_y = state_for_llm['snake'][0]
    dx, dy = dxy_mapping[action]
    new_head = (head_x + dx, head_y + dy)
    if new_head in state_for_llm['snake'] or not (0 < new_head[0] < 7 and 0 < new_head[1] < 7):
        return False, 0
    new_state_for_llm = {
        'snake': [new_head] + deepcopy(state_for_llm['snake'][:-1])
    }
    # validity in X steps
    q = queue.Queue()
    q.put((new_state_for_llm, 0))  # (state, steps)
    while not q.empty():
        current_state, steps = q.get()
        for next_action in range(1, 5):  # Actions: 1 (L), 2 (R), 3 (D), 4 (U)
            next_dx, next_dy = dxy_mapping[next_action]
            next_head = (current_state['snake'][0][0] + next_dx, current_state['snake'][0][1] + next_dy)
            if next_head in current_state['snake'] or not (0 < next_head[0] < 7 and 0 < next_head[1] < 7):
                continue
            new_snake = [next_head] + deepcopy(current_state['snake'][:-1])
            new_state = {
                'snake': new_snake,
            }
            if steps + 1 >= 5:  # If it takes more than 5 steps, it's invalid
                return True, 0
            q.put((new_state, steps + 1))
    return False, steps + 1

def bfs(state_for_llm, step = 8):
    q = queue.Queue()
    q.put((state_for_llm, "", 0))  # (state, action_sequence, reward)
    max_reward = -float('inf')
    best_actions = ""
    while not q.empty():
        current_state, actions, reward = q.get()
        # print(current_state, actions, reward)
        if len(actions) >= step:  # Limit to 8 actions
            if reward > max_reward:
                max_reward = reward
                best_actions = actions
            continue
        for action in range(1, 5):  # Actions: 1 (L), 2 (R), 3 (D), 4 (U)
            dx, dy = dxy_mapping[action]
            new_head = (current_state['snake'][0][0] + dx, current_state['snake'][0][1] + dy)
            if new_head in current_state['snake'] or not (0 < new_head[0] < 7 and 0 < new_head[1] < 7):
                continue
            new_snake = [new_head] + deepcopy(current_state['snake'][:-1])
            new_food = []
            new_reward = reward
            for (x, y, food_value, life_span) in current_state['foods']:
                if (x, y) == new_head:
                    new_reward += food_value
                    if food_value > 0:
                        new_snake.append(current_state['snake'][-1])
                elif life_span > 1:
                    new_food.append((x, y, food_value, life_span - 1))
            next_state = {
                'snake': new_snake,
                'foods': new_food,
            }
            q.put((next_state, actions + dir_mapping[action], new_reward))
    return best_actions, max_reward
def generate_dataset():        
    """
    1. randomly generate a snake
    2. test validity of neighbouring cells
        a) If the snake head goes to the neighbouring cell, it must die in X steps (Invalid, X) (X = 1...5)
        b) Otherwise it is a valid move (Valid)
    3. Create circumstances like following:
        In 4 directions, only one/two direction is valid. 
        a) Place +1 food in some invalid direction.
        b) Place -1 food in some valid directions.
    4. Bfs search for best sequence of actions under partial observability.
    5. Save dataset (snake, food, action label, max reward)
    """
    env = Environment('snake', sticky_action_prob=0)
    logs = {
        "snake": [],
        "dir": [],
        "foods": [],
        "action_label": [],
        "best_actions": [],
        "max_reward": [],
        "render": [],
    }
    for seed in range(1000):
        print(f"Generating dataset for seed {seed}")
        env.seed(seed)
        env.env.reset(initialize_food=False)
        rnd = env.env.random
        snake_len = rnd.randint(6, 15)
        snake = [(rnd.randint(1, 7), rnd.randint(1, 7))]
        for _ in range(snake_len):
            choices = []
            for dir in range(1, 5):  # Actions: 1 (L), 2 (R), 3 (D), 4 (U)
                dx, dy = dxy_mapping[dir]
                new_pos = (snake[-1][0] + dx, snake[-1][1] + dy)
                if new_pos not in snake and 0 < new_pos[0] < 7 and 0 < new_pos[1] < 7:
                    choices.append(dir)
            if not choices:
                break
            dx, dy = dxy_mapping[rnd.choice(choices)]
            snake.append((snake[-1][0] + dx, snake[-1][1] + dy))
        if len(snake) < 6:
            continue
        head = snake[-1]
        body = snake[:-1]
        dx, dy = head[0] - body[-1][0], head[1] - body[-1][1]
        dir = 1 if dy == -1 else 2 if dx == -1 else 3 if dy == 1 else 4
        env.env.snake = deepcopy(snake)
        env.env.dir = dir
        head = (head[1], 7 - head[0])  # Convert to (x, y) format
        # check validity of neighbouring cells
        valid_moves = [0] * 5  # index 0 is unused
        state_for_llm = llm_state_builder(env.env)
        has_false = False
        has_True = False
        for a in range(1, 5):
            dx, dy = dxy_mapping[a]
            valid_moves[a] = check_valid_move(state_for_llm, a)
            # print(f"Action {a}: {valid_moves[a]}")
            # print(f"Head: {head}, Direction: {dir}, Validity: {valid_moves[a]}")
            if valid_moves[a][0] == False and valid_moves[a][1] > 0:
                has_false = True
                if rnd.rand() > 0.2:
                    # place +1 food in some invalid direction
                    env.env.spawn_food((7 - (head[1] + dy),head[0] + dx,  rnd.randint(5, 21), 1))
            if valid_moves[a][0] == True:
                has_True = True
                if rnd.rand() > 0.6:
                    # place -1 food in some valid directions
                    env.env.spawn_food((7 - (head[1] + dy),head[0] + dx,  rnd.randint(5, 21), -1))
        for _ in range(3):
            # place some random food
            x = rnd.randint(1, 7)
            y = rnd.randint(1, 7)
            if (x, y) not in snake and abs(x - snake[-1][0])+ abs(y - snake[-1][1]) < 5:
                env.env.spawn_food((x, y, rnd.randint(5, 21), rnd.choice([-1, 1])))
        # bfs search for best sequence of actions under partial observability
        if not (has_false and has_True):
            continue
        state_for_llm = llm_state_builder(env.env)

        print("Valid seeds:", seed, "Valid moves:", valid_moves)
        best_actions, max_reward = bfs(state_for_llm)
        logs["snake"].append(env.env.snake)
        logs["dir"].append(env.env.dir)
        foods = []
        for (x, y) in env.env.food:
            foods.append((x, y, env.env.food_attributes[x][y][0], env.env.food_attributes[x][y][1]))
        logs["foods"].append(foods)
        logs["action_label"].append(valid_moves)
        logs["best_actions"].append(best_actions)
        logs["max_reward"].append(max_reward)
        logs["render"].append("\n"+env.env.state_string()+"\n")
    # Save dataset
    df = pd.DataFrame(logs)
    df.to_csv('data/snake/dataset.csv', index=False)
        

if __name__ == "__main__":
    generate_dataset()
