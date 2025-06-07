import numpy as np
ROWS = 5
COLS = 7
PLANT_COST = {
    'X': 50,    # 向日葵
    'W': 100,   # 豌豆
    'S': 325,   # 三线豌豆
    'J': 50,    # 坚果
    'H': 125,   # 火炬
    'F': 300    # 辣椒（火爆辣椒）
}
PLANT_HEALTH = {
    'X': 2,
    'W': 2,
    'S': 2,
    'J': 10,
    'H': 2,
    'F': float("inf")
}
ZOMBIE_HEALTH_AND_ATTACK = {
    'normal': {"health": 4, "attack": 1}, 
    'roadblock': {"health": 8, "attack": 1}, 
    'barrel': {"health": 12, "attack": 1}, 
    'high': {"health": 6, "attack": 3}
}
ZOMBIE_RENDER = {
    'normal': "N",
    'roadblock': "R", 
    'barrel': "B", 
    'high': "I"
}
class Env:
    def __init__(self, ramping=None, difficulty = 8):
        self.action_map = ['n','l','u','r','d','f']
        self.random = np.random.RandomState()
    def reset(self):
        self.board = {
            "plants": {},  # {(row, col): {"type": str, "health": int}}
            "zombies": [],  # [{'type': str, 'row': int, 'col': int, 'health': int, 'attack': int}]
            "sun": 50,
            "score": 0,
            "game_over": 0,
            "total_rounds": 0
        }
    def act(self, action_str):
        actions = []
        for cmd in action_str.split(';'):
            cmd = cmd.strip()
            if not cmd:
                continue
            parts = cmd.split()
            if len(parts) != 3:
                continue
            p_type = parts[0].upper()
            try:
                row = int(parts[1])
                col = int(parts[2])
            except ValueError:
                continue
            if p_type not in PLANT_COST:
                continue
            if not (0 <= row < ROWS) or not (0 <= col < COLS):
                continue
            actions.append((p_type, row, col))
        # Each round consists of:
        # 0. Player action (place a plant or do nothing)
        # 1. Cleanup (remove dead plants and zombies)
        # 2. Plants act (shoot or block)
        # 3. Zombies act (move and attack)
        # 4. New zombies are generated, sun accumulates
        self.process_action(actions)
        self.cleanup()
        self.plants_action()
        self.zombies_action()
        self.generate_new_zombies()
        self.board["sun"] += 25
        self.board['total_rounds'] += 1
        self.board['score'] += 1
        return 1, self.check_game_over()

    def process_action(self, action):
        # action is list of (plant_type, row, col)
        for plant in action:
            plant_type, row, col = plant
            # Check if the position is valid
            if (row, col) in self.board["plants"]:
                continue
            # Check if the sun is sufficient for the plant type
            if self.board["sun"] >= PLANT_COST.get(plant_type, float("inf")):
                self.board["sun"] -= PLANT_COST[plant_type]
                self.board["plants"][(row, col)] = {
                    "type": plant_type,
                    "health": PLANT_HEALTH[plant_type]
                }
            else:
                continue

    def cleanup(self):
        self.board["zombies"] = [zombie for zombie in self.board["zombies"] if zombie["health"] > 0]
        self.board["plants"] = {pos: data for pos, data in self.board["plants"].items() if data["health"] > 0}

    def state_string(self):
        board = self.board
        header = f"Turn:{board['total_rounds']} | Sun:{board['sun']} | Score: {board['score']}"
        grid = [['0'] * COLS for _ in range(ROWS)]
        # place plants
        for (r, c), plant in board["plants"].items():
            grid[r][c] = plant["type"]
        # place zombies
        for zombie in board["zombies"]:
            r, c = zombie["row"], zombie["col"]
            z_char = ZOMBIE_RENDER.get(zombie["type"], "?")
            if grid[r][c] == '0':
                grid[r][c] = z_char
            else:
                grid[r][c] += z_char
        grid_str = "\n".join([f"Row {i}|" + "|".join(f"{cell:3}" for cell in row) for i, row in enumerate(grid)])
        return header + "\n" + grid_str
    def plants_action(self):
        self.chilli_action()
        self.sun_flower_action()
        self.peas_action()
    
    def chilli_action(self):
        # 火爆辣椒：将所在行所有僵尸清除，同时自身消失
        for pos in list(self.board["plants"].keys()):
            plant_type = self.board["plants"][pos]["type"]
            if plant_type == "F":
                row, _ = pos
                self.board["zombies"] = [zombie for zombie in self.board["zombies"] if zombie["row"] != row]
                del self.board["plants"][pos]
    
    def sun_flower_action(self):
        # 向日葵：每个向日葵增加 10 点阳光
        for pos in list(self.board["plants"].keys()):
            plant_type = self.board["plants"][pos]["type"]
            if plant_type == "X":
                self.board["sun"] += 10
    
    def peas_action(self):
        # 豌豆类植物攻击：W 为单线攻击，S 为三线攻击
        peas = [(pos, data) for pos, data in self.board['plants'].items() if data["type"] in ('W', 'S')]
        for pos, data in peas:
            row, col = pos
            if data["type"] == "W":
                lines = [row]
            elif data["type"] == "S":
                lines = [max(0, row - 1), row, min(ROWS - 1, row + 1)]
            else:
                lines = []
            
            for line in lines:
                zombies_in_line = [zombie for zombie in self.board["zombies"] if zombie["row"] == line]
                if not zombies_in_line:
                    continue
                # 选择离植物最近的僵尸（即列值最小的僵尸）
                target_zombie = min(zombies_in_line, key=lambda x: x["col"])
                dmg = self.calculate_damage(line, col)
                idx = self.board["zombies"].index(target_zombie)
                target_zombie["health"] -= dmg
                if target_zombie["health"] <= 0:
                    del self.board["zombies"][idx]
                else:
                    self.board["zombies"][idx] = target_zombie
    
    def calculate_damage(self, row, col):
        base_damage = 1
        # 若在植物和僵尸之间存在火炬植物，则增加额外伤害
        for c in range(col, COLS):
            if (row, c) in self.board["plants"]:
                if self.board["plants"][(row, c)]["type"] == "H":
                    base_damage += 1
                    break
        return base_damage
    
    def zombies_action(self):
        new_zombies = []
        for zombie in self.board["zombies"]:
            row = zombie["row"]
            col = zombie["col"]
            attack = zombie["attack"]
            # 如果当前位置有植物，则攻击植物
            if (row, col) in self.board["plants"]:
                plant = self.board["plants"][(row, col)]
                new_health = plant["health"] - attack
                if new_health <= 0:
                    del self.board["plants"][(row, col)]
                else:
                    self.board["plants"][(row, col)]["health"] = new_health
                new_zombies.append(zombie)
            else:
                new_col = col - 1
                if new_col < 0:
                    # 僵尸越界，游戏结束
                    self.board["score"] = self.board["total_rounds"]
                    self.board["game_over"] = 1
                    return
                zombie["col"] = new_col
                new_zombies.append(zombie)
        self.board["zombies"] = new_zombies
    
    def generate_new_zombies(self):
        # 每 5 回合生成僵尸，每 10 回合增加一个僵尸
        round_num = self.board["total_rounds"]
        if round_num % 5 != 0:
            return
        base_count = 1 + int(round_num // 10)
        for _ in range(base_count):
            row = self.random.randint(0, ROWS - 1)
            zombie_type = self.select_zombie_type(round_num)
            health = ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["health"] + (round_num // 10) * 4
            self.board["zombies"].append({
                "type": zombie_type,
                "row": row,
                "col": COLS - 1,
                "health": health,
                "attack": ZOMBIE_HEALTH_AND_ATTACK[zombie_type]["attack"]
            })
    
    def select_zombie_type(self, round_num):
        if round_num < 10:
            return "normal"
        elif round_num < 20:
            return self.random.choice(["normal", "roadblock"])
        else:
            return self.random.choice(["normal", "roadblock", "barrel", "high"])
    
    def check_game_over(self):
        if self.board["game_over"] == 1:
            return 1
        if self.board["total_rounds"] >= 100:
            return 1
        return 0
