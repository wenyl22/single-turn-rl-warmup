import numpy as np
from copy import deepcopy
class Env:
    def __init__(self, ramping=None):
        self.random = np.random.RandomState()
        self.seed = 2042

    def reset(self):
        self.B = 8 if 2000 <= self.seed < 3000 else 6 if 1000 <= self.seed < 2000 else 10
        self.snake = [(self.B // 2 - 1, self.B // 2 - 1)]
        self.food = []
        self.food_attributes = [[0 for _ in range(self.B)] for _ in range(self.B)]

        self.dir = 'L'
        self.game_turn = 0
        self.reward = 0
        self.terminal = False
        # random permute coords
        self.coords = [(x, y) for x in range(1, self.B - 1) for y in range(1, self.B - 1)]
        self.random.shuffle(self.coords)
        self.random.shuffle(self.coords)
        # random choose 30% of index in range(200) and set self.value to -1
        self.value = [1 for _ in range(200)]
        lst = [_ for _ in range(200)]
        self.random.shuffle(lst)
        chosen_index = lst[:60]
        for idx in chosen_index:
            self.value[idx] = -1
        self.idx = 0
        self.spawn_food()
        self.spawn_food()
        self.spawn_food()

    def spawn_food(self):
        flag = [(x, y) in self.snake or self.food_attributes[x][y] != 0 for (x, y) in self.coords]
        if sum(flag) >= len(self.coords) - 1:
            return
        while flag[self.idx]:
            self.idx += 1
            if self.idx >= len(self.coords):
                self.idx -= len(self.coords)

        x, y = self.coords[self.idx]
        self.idx += 1
        if self.idx >= len(self.coords):
            self.idx -= len(self.coords)
        life_span = 12
        value = self.value.pop(0)
        new_food = (x, y)
        assert self.food_attributes[x][y] == 0 and new_food not in self.food, \
            f"Food already exists at {new_food}, attributes: {self.food_attributes[x][y]}, coords: {self.coords}"
        self.food.append(new_food)
        self.food_attributes[x][y] = (life_span, value)

    def act(self, a):
        self.r = 0
        self.game_turn += 1
        if (a == 'L' and self.dir == 'R') or \
            (a == 'R' and self.dir == 'L') or \
            (a == 'U' and self.dir == 'D') or \
            (a == 'D' and self.dir == 'U'):
                a = self.dir # prevent reverse direction
                raise ValueError(f"Invalid action a = {a}, dir = {self.dir}")
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
            self.r -= 1
            self.reward += self.r
            self.terminal = True
            return self.r, self.terminal
        self.snake.append(new_head)

        if new_head in self.food:
            self.r += self.food_attributes[x][y][1]
            self.food.remove(new_head)
            self.food_attributes[x][y] = 0
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
        if self.game_turn % 2 == 1:
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
        new_env.random = deepcopy(self.random)
        new_env.snake = deepcopy(self.snake)
        new_env.food = deepcopy(self.food)
        new_env.food_attributes = deepcopy(self.food_attributes)
        new_env.dir = self.dir
        new_env.game_turn = self.game_turn
        new_env.reward = self.reward
        new_env.terminal = self.terminal
        new_env.coords = self.coords.copy()
        new_env.value = self.value.copy()
        new_env.B = self.B
        new_env.seed = self.seed
        return new_env
    def get_possible_actions(self):
        # return 'L', 'R', 'U', 'D' removing the reverse of the current direction
        if self.dir == 'L':
            actions = ['L', 'U', 'D']
        elif self.dir == 'R':
            actions = ['R', 'U', 'D']
        elif self.dir == 'U':
            actions = ['L', 'R', 'U']
        elif self.dir == 'D':
            actions = ['L', 'R', 'D']
        return actions