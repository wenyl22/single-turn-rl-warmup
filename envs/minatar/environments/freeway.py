################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian (ttian@ualberta.ca)                                                                                #
################################################################################################################
import numpy as np


#####################################################################################################################
# Constants
#
#####################################################################################################################
player_speed = 1
time_limit = 2500


#####################################################################################################################
# Env
#
# The player begins at the bottom of the screen and motion is restricted to traveling up and down. Player speed is
# also restricted such that the player can only move every 3 frames. A reward of +1 is given when the player reaches
# the top of the screen, at which point the player is returned to the bottom. Cars travel horizontally on the screen
# and teleport to the other side when the edge is reached. When hit by a car, the player is returned to the bottom of
# the screen. Car direction and speed is indicated by 5 trail channels, the location of the trail gives direction
# while the specific channel indicates how frequently the car moves (from once every frame to once every 5 frames).
# Each time the player successfully reaches the top of the screen, the car speeds are randomized. Termination occurs
# after 2500 frames have elapsed.
#
#####################################################################################################################
class Env:
    def __init__(self, ramping=None, difficulty = 8):
        self.channels ={
            'chicken':0,
            'car':1,
            'speed1':2,
            'speed2':3,
            'speed3':4,
            'speed4':5,
            'speed5':6,
        }
        self.difficulty = difficulty
        self.action_map = ['n','l','u','r','d','f']
        self.random = np.random.RandomState()
        self.special_pattern = False
    # Reset to start state for new episode
    def reset(self):
        # choose self.difficulty numbers in [1, 9] to be freeways with cars
        self.chosen_freeways = self.random.choice(range(0, 8), self.difficulty, replace=False)
        self.chosen = [True if i in self.chosen_freeways else False for i in range(8)]
        self._randomize_cars(initialize=True)
        self.pos = 9
        self.move_timer = player_speed - 1
        self.terminate_timer = time_limit
        self.terminal = False

    # Randomize car speeds and directions, also reset their position if initialize=True
    def _randomize_cars(self, initialize=False):
        directions = np.sign(self.random.rand(8) - 0.5).astype(int)
        self.cars = []
        # if not self.special_pattern:
        #     speeds = self.random.randint(1, 5, 8) * directions
        #     for i in range(8):
        #         self.cars+=[[0, i+1, abs(speeds[i]) - 1,speeds[i], 1]]
        #     return
        # Special Patterns: 
        # 1. Random batch neighbour lanes, share same car distribution
        # 2. Each car distribution is one of with equal probability:
            # a. Long cars, fast speed.
            # b. Consecutive car fleet, with constant speed and spacing.
            # c. Random one car as original.
        batch = self.random.randint(0, 2, 8)
        cur_cars = []
        for i in range(8):
            if batch[i] == 1 and i != 0:
                for j in range(len(cur_cars)):
                    cur_cars[j][1] = i + 1
                self.cars.extend([car for car in cur_car] for cur_car in cur_cars)
                continue
            cur_cars = []
            rnd = self.random.randint(0, 3) # [0, 2]
            pos = 0 if directions[i] == 1 else 8
            if rnd == 0:
                speed = self.random.randint(2, 5) # [2, 4] 
                length = speed
                cur_cars = [[pos, i + 1, 0, 1.0/speed * directions[i], speed]]
            elif rnd == 1:
                num = self.random.randint(2, 4) # [2, 5]
                speed = self.random.randint(1, 4) # [1, 3]
                for j in range(num):
                    cur_cars.append([pos, i + 1, abs(speed) - 1, speed, 1])
                    pos += 9 // num if directions[i] == 1 else -(9 // num)
                    pos = (pos + 9) % 9
            else:
                speed = self.random.randint(1, 5) #[1, 4]
                cur_cars = [[pos, i + 1, abs(speed) - 1, speed, 1]]
            self.cars.extend([car for car in cur_car] for cur_car in cur_cars)
        for i in range(len(self.cars)):
            if self.chosen[self.cars[i][1] - 1] == False:
                self.cars[i] = [None, self.cars[i][1], None, None, None]
    # Update environment according to agent action
    def act(self, a):
        r = 0
        if(self.terminal):
            return r, self.terminal
        if type(a) is not str:
            a = self.action_map[a]

        if(a=='u' and self.move_timer==0):
            self.move_timer = player_speed
            self.pos = max(0, self.pos-1)
        elif(a=='d' and self.move_timer==0):
            self.move_timer = player_speed
            self.pos = min(9, self.pos+1)

        # Win condition
        if(self.pos==0):
            r += 1
            self._randomize_cars(initialize=False)
            self.pos = 9

        # Update cars
        # car: [x, y, timer, speed, length]
        for car in self.cars:
            if car[3] is None:
                continue
            dir = -1 if car[3] > 0 else 1
            # # collision check
            # for l in range(car[4]):
            #     if car[0] + l*dir == 4 and self.pos == car[1]:
            #         self.pos = 9
            #         r = -1
            if(car[0:2] == [4, self.pos]):
                self.pos = 9
            if(car[0] < 0):
                car[0] = 8
            elif(car[0] > 8):
                car[0] = 0
            else:
                if(abs(car[3]) >= 1):
                    car[2] -= 1
                    if car[2] == -1:
                        car[2] += abs(car[3])
                        car[0] += 1 if car[3] > 0 else -1
                else:
                    car[0] += int(1/car[3])
            # collision check
            for l in range(car[4]):
                if car[0] + l*dir == 4 and self.pos == car[1]:
                    self.pos = 9
                    r = -1

        # Update various timers
        self.move_timer -= self.move_timer > 0
        self.terminate_timer -= 1
        if(self.terminate_timer < 0):
            self.terminal = True
        return r, self.terminal

    # Query the current level of the difficulty ramp, difficulty does not ramp in this game, so return None
    def difficulty_ramp(self):
        return None

    # Process the game-state into the 10x10xn state provided to the agent and return
    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)
        state[self.pos, 4, self.channels['chicken']] = 1
        for car in self.cars:
            if car[3] is None:
                continue
            state[car[1],car[0], self.channels['car']] = 1
            back_x = car[0]-1 if car[3]>0 else car[0]+1
            if(back_x<0):
                back_x=9
            elif(back_x>9):
                back_x=0
            if(abs(car[3])==1):
                trail = self.channels['speed1']
            elif(abs(car[3])==2):
                trail = self.channels['speed2']
            elif(abs(car[3])==3):
                trail = self.channels['speed3']
            elif(abs(car[3])==4):
                trail = self.channels['speed4']
            elif(abs(car[3])==5):
                trail = self.channels['speed5']
            state[car[1],back_x, trail] = 1
        return state

    # Dimensionality of the game-state (10x10xn)
    def state_shape(self):
        return [10,10,len(self.channels)]

    # Subset of actions that actually have a unique impact in this environment
    def minimal_action_set(self):
        minimal_actions = ['n','u','d']
        return [self.action_map.index(x) for x in minimal_actions]
    def from_dict(self, d):
        self.pos = d['pos']
        self.move_timer = d['move_timer']
        self.terminate_timer = time_limit
        self.terminal = False
        self.cars = d['cars']
        
    def state_string(self):
        grid_string = ""
        for i in range(10): # rows, i.e. y
            for j in range(9): # columns, i.e. x
                grid_string_add = ""
                if(j == 4 and self.pos == i):
                    grid_string_add += 'P'
                for car in self.cars:
                    if car[3] is None or car[1] != i:
                        continue
                    dir = 1 if car[3] > 0 else -1
                    if(car[0] == j):
                        speed = abs(car[3])
                        speed = '/'+str(speed) if speed >= 1 else str(int(abs(1/car[3])))
                        if car[3] > 0:
                            grid_string_add += speed + '>'
                        else:
                            grid_string_add = '<' + speed
                    else:
                        if car[0] < j < car[0] - dir * car[4]:
                            grid_string_add += 'x'
                        if car[0] - dir * car[4] < j < car[0]:
                            grid_string_add += 'x'
                if grid_string_add == "":
                    grid_string_add = "."
                grid_string += grid_string_add                            
                grid_string += "".join([" "] * (4 - len(grid_string_add)))
                grid_string += " "
            grid_string += "\n"
        return grid_string
    def deep_copy(self):
        env = Env()
        env.channels = self.channels
        env.difficulty = self.difficulty
        env.action_map = self.action_map
        env.random = self.random
        env.chosen_freeways = self.chosen_freeways
        env.chosen = self.chosen
        env.cars = [[car for car in cars] for cars in self.cars]
        env.pos = self.pos
        env.move_timer = self.move_timer
        env.terminate_timer = self.terminate_timer
        env.terminal = self.terminal
        return env
