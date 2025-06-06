import numpy as np
from copy import deepcopy
board_size = 8
class Env:
    def __init__(self, ramping=None, difficulty = 8):
        self.action_map = ['n','l','u','r','d','f']
        self.random = np.random.RandomState()
    def reset(self, initialize_food=True):
        self.initial_map = [[0 for _ in range(board_size)] for _ in range(board_size)]
        self.snake = [(board_size // 2, board_size // 2 - 1)]
        self.food = []
        self.food_attributes = deepcopy(self.initial_map)
        self.dir = 1
        self.wall = []
        if initialize_food:
            self.spawn_food()
    def spawn_food(self, tple=None):
        if tple is None:
            while len(self.food) < 3:
                x = self.random.randint(1, board_size - 2)
                y = self.random.randint(1, board_size - 2)
                new_food = (x, y)
                if new_food not in self.snake and new_food not in self.wall and new_food not in self.food:
                    value = -1
                    if self.random.rand() < 0.9:
                        value = 1
                    life_span = self.random.randint(5, 21)
                    self.food.append(new_food)
                    self.food_attributes[x][y] = (life_span, value)
        else:
            x, y, life_span, value = tple
            new_food = (x, y)
            self.food.append(new_food)
            self.food_attributes[x][y] = (life_span, value)
    def act(self, a):
        r = 0
        if (a == 1 and self.dir == 3) \
              or (a == 2 and self.dir == 4) \
              or (a == 3 and self.dir == 1) \
              or (a == 4 and self.dir == 2):
                a = self.dir # prevent reverse direction
        if 1 <= a <= 4: # ignore invalid actions
            self.dir = a
        head_x, head_y = self.snake[-1]
        if self.dir == 1:
            new_head = (head_x, head_y - 1)
        elif self.dir == 3:
            new_head = (head_x, head_y + 1)
        elif self.dir == 2:
            new_head = (head_x - 1, head_y)
        elif self.dir == 4:
            new_head = (head_x + 1, head_y)
        else:
            raise ValueError(f"Invalid action a = {a}, dir = {self.dir}")     
        x, y = new_head
        if new_head in self.snake[1:] or new_head in self.wall or \
        new_head[0] == 0 or new_head[1] == 0 or \
        new_head[0] == board_size - 1 or new_head[1] == board_size - 1:
            return -1, True
        self.snake.append(new_head)
        if new_head in self.food:
            r = self.food_attributes[x][y][1]
            self.food.remove(new_head)
            self.spawn_food()
            if r < 0:
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
        return r, False

    def state_string(self):
        grid_string = ""
        l = len(self.snake)
        for i in range(board_size):
            for j in range(board_size):
                output = ""
                if (i, j) in self.snake:
                    output += chr(ord('a') + l - 1 - self.snake.index((i, j)))
                elif (i, j) in self.food:
                    if self.food_attributes[i][j][1] > 0:
                        output += '+'
                    else:
                        output += '-'
                    output += str(self.food_attributes[i][j][0])
                elif i == 0 or i == board_size - 1 or j == 0 or j == board_size - 1 or (i, j) in self.wall:
                    output += '#'
                else:
                    output += '.'
                grid_string += output + ' ' * (4 - len(output))
            grid_string += '\n'
        return grid_string

    def deep_copy(self):
        new_env = Env()
        new_env.channels = self.channels.copy()
        new_env.action_map = self.action_map.copy()
        new_env.random = self.random
        new_env.initial_map = deepcopy(self.initial_map)
        new_env.snake = deepcopy(self.snake)
        new_env.food = deepcopy(self.food)
        new_env.wall = deepcopy(self.wall)
        return new_env