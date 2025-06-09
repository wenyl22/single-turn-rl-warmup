import numpy as np
from copy import deepcopy
class Env:
    def __init__(self, ramping=None):
        self.action_map = ['n','l','u','r','d','f']
        self.random = np.random.RandomState()
    def reset(self):
        self.pos = 4
        self.space_ships = [(i, -1, 1, 0) for i in range(10)]
        self._randomize_spaceships()
        self.reward = 0
        self.game_turn = 0
        self.terminal = False
        # space ship representation(x, y, speed, reward)

    def _randomize_spaceships(self):
        for _ in range(4):
            temp = [i for i in range(10) if self.space_ships[i][1] <= 0]
            if len(temp) == 0:
                return
            index = self.random.choice(temp)
            speed = self.random.randint(1, 6)
            reward = self.random.randint(1, 25)
            self.space_ships[index] = (index, 9, speed, reward)
        
    # Update environment according to agent action
    def act(self, a):
        self.r = 0
        self.game_turn += 1
        # Move the player
        if a == 'L':
            self.pos = max(0, self.pos - 1)
        elif a == 'R':
            self.pos = min(9, self.pos + 1)
        # Move the space ships and check for collisions
        for i in range(10):
            x, y, speed, reward = self.space_ships[i]
            if y <= 0:
                continue
            y -= speed
            if y <= 0:
                # print(f"Collision at {x}, {y} with speed {speed} and reward {reward}")
                if x == self.pos:
                    self.r += reward
                reward = 0
                y = -1
            self.space_ships[i] = (x, y, speed, reward)
        self._randomize_spaceships()
        self.reward += self.r
        self.terminal = True if self.game_turn >= 100 else False
        return self.r, self.terminal
             
    def difficulty_ramp(self):
        return None

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        return None

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n', 'l', 'r']
        return [self.action_map.index(x) for x in minimal_actions]
        
    def state_string(self):
        grid_string = ""
        for i in range(10):
            for j in range(10):
                grid_string_add = ""
                if self.space_ships[j][1] == 9 - i:
                    grid_string_add = str(self.space_ships[j][2]) + ' ' + str(self.space_ships[j][3])
                elif i == 9 and self.pos == j:
                    grid_string_add += "P"
                else:
                    grid_string_add += "."
                grid_string += grid_string_add + "".join([" "] * (6 - len(grid_string_add)))
            grid_string += "\n"
        assert "P" in grid_string, "Player position not found in grid string"
        return grid_string
    def deep_copy(self):
        new_env = Env()
        new_env.channels = self.channels.copy()
        new_env.action_map = self.action_map.copy()
        new_env.random = self.random
        new_env.spawn_timer = self.spawn_timer
        new_env.pos = self.pos
        new_env.space_ships = deepcopy(self.space_ships)
        return new_env
