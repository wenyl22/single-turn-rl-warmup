import numpy as np
from copy import deepcopy
class Env:
    def __init__(self, ramping=None):
        self.action_map = ['n','l','u','r','d','f']
        self.random = np.random.RandomState()
        self.B = 8
    def reset(self):
        self.initial_map = [[0 for _ in range(self.B)] for _ in range(self.B)]
        self.snake = [(self.B // 2, self.B // 2 - 1)]
        self.food = []
        self.food_attributes = deepcopy(self.initial_map)
        self.dir = 'L'
        self.game_turn = 0
        self.reward = 0
        self.terminal = False
        self.spawn_food()
    def spawn_food(self):
        self.empty_coords = [(x, y) for x in range(1, self.B - 1) for y in range(1, self.B - 1)]
        for (x, y) in self.snake:
            self.empty_coords.remove((x, y))
        for (x, y) in self.food:
            self.empty_coords.remove((x, y))
        while len(self.food) < 3 and len(self.empty_coords) > 0:
            idx = self.random.randint(len(self.empty_coords))
            x, y = self.empty_coords[idx]
            new_food = (x, y)
            if new_food not in self.snake and new_food not in self.food:
                self.food.append(new_food)
                life_span = self.random.randint(5, 21)
                value = -1
                if self.random.rand() < 0.7:
                    value = 1
                self.food_attributes[x][y] = (life_span, value)
    def act(self, a):
        self.r = 0
        self.game_turn += 1
        if (a == 'L' and self.dir == 'R') or \
            (a == 'R' and self.dir == 'L') or \
            (a == 'U' and self.dir == 'D') or \
            (a == 'D' and self.dir == 'U'):
                a = self.dir # prevent reverse direction
        if a in ['L', 'R', 'U', 'D']: # ignore invalid actions
            self.dir = a
        head_x, head_y = self.snake[-1]
        if self.dir == 'L':
            new_head = (head_x - 1, head_y)
        elif self.dir == 'R':
            new_head = (head_x + 1, head_y)
        elif self.dir == 'D':
            new_head = (head_x, head_y - 1)
        elif self.dir == 'U':
            new_head = (head_x, head_y + 1)
        else:
            raise ValueError(f"Invalid action a = {a}, dir = {self.dir}")     
        x, y = new_head
        if new_head in self.snake[1:] or \
        new_head[0] == 0 or new_head[1] == 0 or \
        new_head[0] == self.B - 1 or new_head[1] == self.B - 1:
            self.r = -1
            self.reward += self.r
            self.terminal = True
            return self.r, self.terminal
        self.snake.append(new_head)
        if new_head in self.food:
            self.r = self.food_attributes[x][y][1]
            self.food.remove(new_head)
            self.spawn_food()
            if self.r < 0:
                self.snake.pop(0)
        else:
            self.snake.pop(0)
        for food in self.food:
            x, y = food
            l, v = self.food_attributes[x][y]
            self.food_attributes[x][y] = (l - 1, v)
            if l <= 1:
                self.food.remove(food)
                self.food_attributes[x][y] = 0
                self.spawn_food()
        self.reward += self.r
        self.terminal = True if self.game_turn >= 100 else False
        return self.r, self.terminal

    def state_string(self):
        grid_string = ""
        l = len(self.snake)
        for i in range(self.B):
            for j in range(self.B):
                output = ""
                x, y = j, self.B - 1 - i
                if (x, y) in self.snake:
                    output += chr(ord('a') + l - 1 - self.snake.index((x, y)))
                elif (x, y) in self.food:
                    if self.food_attributes[x][y][1] > 0:
                        output += '+'
                    else:
                        output += '-'
                    output += str(self.food_attributes[x][y][0])
                elif x == 0 or x == self.B - 1 or y == 0 or y == self.B - 1:
                    output += '#'
                else:
                    output += '.'
                grid_string += output + ' ' * (4 - len(output))
            grid_string += '\n'
        return grid_string

    def deep_copy(self):
        new_env = Env()
        new_env.action_map = self.action_map.copy()
        new_env.random = self.random
        new_env.initial_map = deepcopy(self.initial_map)
        new_env.snake = deepcopy(self.snake)
        new_env.food = deepcopy(self.food)
        return new_env