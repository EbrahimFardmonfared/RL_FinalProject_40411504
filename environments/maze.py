import numpy as np
import random
import copy
from collections import deque

class DynamicMazeEnv:
    EMPTY = 0
    WALL = 1
    PENALTY = 2
    KEY = 3
    DOOR = 4
    GOAL = 5

    def __init__(self, use_reward_shaping=False):
        self.use_reward_shaping = use_reward_shaping
        self.grid_size = 15
        self.seed = 0
        
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.start_pos = (0, 0)
        self.key_pos = (2, 12)
        self.goal_pos = (14, 14)
        self.door_pos = (13, 14) # در دقیقاً بالای خانه هدف قرار دارد
        
        self.patrol_route = [(7, 4), (7, 5), (7, 6), (8, 6), (9, 6), (9, 5), (9, 4), (8, 4)]
        
        self._generate_valid_map()
        self.reset()

    def _generate_valid_map(self):
        max_attempts = 1000
        for _ in range(max_attempts):
            self._build_grid()
            if self._bfs_check():
                return
        raise Exception("Could not generate a valid map after 1000 attempts.")

    def _build_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        self.grid[self.key_pos] = self.KEY
        self.grid[self.door_pos] = self.DOOR
        self.grid[self.goal_pos] = self.GOAL
        
        # مسدود کردن سمت چپ هدف برای تبدیل کردن در به تنها گلوگاه ورودی
        self.grid[14, 13] = self.WALL 
        
        cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)]
        cells.remove(self.start_pos)
        cells.remove(self.key_pos)
        cells.remove(self.door_pos)
        cells.remove(self.goal_pos)
        cells.remove((14, 13)) # حذف از لیست خانه‌های خالی
        for p in set(self.patrol_route):
            if p in cells:
                cells.remove(p)
                
        random.shuffle(cells)
        
        # 34 دیوار رندوم + 1 دیوار ثابت (14, 13) = 35 دیوار (بیشتر از 15 درصد)
        for i in range(34):
            self.grid[cells[i]] = self.WALL
        for i in range(34, 42):
            self.grid[cells[i]] = self.PENALTY

    def _bfs_check(self):
        if not self._has_path(self.start_pos, self.key_pos, impassable=[self.WALL, self.DOOR]):
            return False
        if not self._has_path(self.key_pos, self.goal_pos, impassable=[self.WALL]):
            return False
        return True

    def _has_path(self, start, target, impassable):
        queue = deque([start])
        visited = set([start])
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == target:
                return True
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) not in visited and self.grid[nr, nc] not in impassable:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def reset(self):
        self.agent_pos = self.start_pos
        self.has_key = False
        self.patrol_idx = 0
        return self._get_state()

    def _get_state(self):
        return (self.agent_pos[0], self.agent_pos[1], int(self.has_key), self.patrol_idx)

    def step(self, action):
        r, c = self.agent_pos
        
        rand_val = random.random()
        if rand_val < 0.8:
            actual_action = action
        elif rand_val < 0.9:
            actual_action = (action - 1) % 4
        else:
            actual_action = (action + 1) % 4
            
        dr, dc = 0, 0
        if actual_action == 0:   dr = -1
        elif actual_action == 1: dc = 1
        elif actual_action == 2: dr = 1
        elif actual_action == 3: dc = -1
        
        nr, nc = r + dr, c + dc
        hit_wall = False
        hit_penalty = False
        
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            nr, nc = r, c
            hit_wall = True
        elif self.grid[nr, nc] == self.WALL:
            nr, nc = r, c
            hit_wall = True
        elif self.grid[nr, nc] == self.DOOR and not self.has_key:
            nr, nc = r, c
            hit_wall = True
            
        self.agent_pos = (nr, nc)
        
        self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_route)
        obstacle_pos = self.patrol_route[self.patrol_idx]
        
        reward = -1.0
        done = False
        
        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True
            reward += 50.0
            
        if self.grid[self.agent_pos] == self.PENALTY:
            reward -= 10.0
            hit_penalty = True
            
        if self.agent_pos == obstacle_pos:
            reward -= 15.0
            hit_penalty = True
            
        if self.agent_pos == self.goal_pos:
            reward += 100.0
            done = True
            
        if hit_wall:
            reward -= 5.0

        if self.use_reward_shaping:
            target_r, target_c = self.goal_pos if self.has_key else self.key_pos
            phi_current = - (abs(r - target_r) + abs(c - target_c))
            
            next_target_r, next_target_c = self.goal_pos if self.has_key else self.key_pos
            phi_next = - (abs(nr - next_target_r) + abs(nc - next_target_c))
            
            gamma = 0.9
            f_reward = gamma * phi_next - phi_current
            reward += f_reward

        info = {'hit_wall': hit_wall, 'hit_penalty': hit_penalty}
        return self._get_state(), reward, done, info

    def get_transitions(self, state, action):
        r, c, has_key, patrol_idx = state
        has_key = bool(has_key)
        transitions = []
        
        next_patrol_idx = (patrol_idx + 1) % len(self.patrol_route)
        obstacle_pos = self.patrol_route[next_patrol_idx]
        
        actions_probs = [
            (action, 0.8),
            ((action - 1) % 4, 0.1),
            ((action + 1) % 4, 0.1)
        ]
        
        for act, prob in actions_probs:
            dr, dc = 0, 0
            if act == 0: dr = -1
            elif act == 1: dc = 1
            elif act == 2: dr = 1
            elif act == 3: dc = -1
            
            nr, nc = r + dr, c + dc
            hit_wall = False
            
            if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                nr, nc = r, c
                hit_wall = True
            elif self.grid[nr, nc] == self.WALL:
                nr, nc = r, c
                hit_wall = True
            elif self.grid[nr, nc] == self.DOOR and not has_key:
                nr, nc = r, c
                hit_wall = True
                
            next_has_key = has_key
            reward = -1.0
            done = False
            
            if (nr, nc) == self.key_pos and not next_has_key:
                next_has_key = True
                reward += 50.0
                
            if self.grid[nr, nc] == self.PENALTY:
                reward -= 10.0
                
            if (nr, nc) == obstacle_pos:
                reward -= 15.0
                
            if (nr, nc) == self.goal_pos:
                reward += 100.0
                done = True
                
            if hit_wall:
                reward -= 5.0
                
            if self.use_reward_shaping:
                target_r, target_c = self.goal_pos if has_key else self.key_pos
                phi_current = - (abs(r - target_r) + abs(c - target_c))
                
                next_target_r, next_target_c = self.goal_pos if next_has_key else self.key_pos
                phi_next = - (abs(nr - next_target_r) + abs(nc - next_target_c))
                
                gamma = 0.9
                reward += (gamma * phi_next - phi_current)
                
            next_state = (nr, nc, int(next_has_key), next_patrol_idx)
            transitions.append((prob, next_state, reward, done))
            
        return transitions

    def generate_similar_map(self):
        max_attempts = 1000
        for _ in range(max_attempts):
            new_env = copy.deepcopy(self)
            new_env.use_reward_shaping = self.use_reward_shaping
            walls = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.WALL]
            if (14, 13) in walls: walls.remove((14, 13)) # محافظت از دیوار مسدودکننده کنار هدف
            empties = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.EMPTY]
            if len(walls) >= 3 and len(empties) >= 3:
                for i in range(3):
                    wr, wc = random.choice(walls)
                    er, ec = random.choice(empties)
                    new_env.grid[wr, wc] = self.EMPTY
                    new_env.grid[er, ec] = self.WALL
                    walls.remove((wr, wc))
                    empties.remove((er, ec))
            if new_env._bfs_check():
                return new_env
        return self

    def generate_different_map(self):
        max_attempts = 1000
        for _ in range(max_attempts):
            new_env = copy.deepcopy(self)
            new_env.use_reward_shaping = self.use_reward_shaping
            walls = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.WALL]
            if (14, 13) in walls: walls.remove((14, 13)) # محافظت از دیوار مسدودکننده کنار هدف
            empties = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.EMPTY]
            random.shuffle(walls)
            random.shuffle(empties)
            for i in range(min(10, len(walls))):
                wr, wc = walls[i]
                er, ec = empties[i]
                new_env.grid[wr, wc] = self.EMPTY
                new_env.grid[er, ec] = self.WALL
            for i in range(10, min(15, len(empties))):
                er, ec = empties[i]
                new_env.grid[er, ec] = self.PENALTY
            if new_env._bfs_check():
                return new_env
        return self