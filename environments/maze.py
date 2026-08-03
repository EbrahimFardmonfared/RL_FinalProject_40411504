import numpy as np
import random
import copy
from collections import deque

class DynamicMazeEnv:
    def __init__(self, use_reward_shaping=False):
        # تنظیمات پایه بر اساس شماره دانشجویی 40411504
        self.seed = 0
        self.grid_size = 15
        
        # سوییچ فعال‌سازی نسخه دوم پاداش (Reward Shaping)
        self.use_reward_shaping = use_reward_shaping
        
        # مقادیر المان‌های نقشه
        self.EMPTY = 0
        self.WALL = 1
        self.PENALTY = 2
        self.START = 3
        self.GOAL = 4
        self.KEY = 5
        self.DOOR = 6
        
        # پارامترهای عدم قطعیت حرکت
        self.PROB_FORWARD = 0.8
        self.PROB_DRIFT = 0.1
        
        # مسیر گشت‌زنی دوبعدی و موجی (حرکت ترکیبی افقی و عمودی که هم بالا/پایین می‌رود و هم حس حرکت پویا/رندوم دارد)
        self.patrol_route = [
            (7, 3), (6, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (8, 10), (7, 11),
            (6, 11), (5, 10), (6, 9), (7, 8), (8, 7), (9, 6), (8, 5), (7, 4)
        ]
        
        self.grid = None
        self._generate_valid_map()
        self.reset()

    def _generate_valid_map(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        is_valid = False
        while not is_valid:
            self._build_grid()
            is_valid = self._bfs_check()
            if not is_valid:
                self.seed += 1 
                np.random.seed(self.seed)
                random.seed(self.seed)

    def _build_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # جایگذاری المان‌های اصلی با ایجاد گلوگاه اجباری برای در
        self.start_pos = (0, 0)
        self.key_pos = (2, 12)
        self.goal_pos = (14, 14)
        self.door_pos = (13, 14) 
        self.choke_wall = (14, 13) 
        
        self.grid[self.start_pos] = self.START
        self.grid[self.key_pos] = self.KEY
        self.grid[self.goal_pos] = self.GOAL
        self.grid[self.door_pos] = self.DOOR
        self.grid[self.choke_wall] = self.WALL 
        
        num_walls = 34
        num_penalties = 6
        
        protected_cells = set([self.start_pos, self.key_pos, self.door_pos, self.goal_pos, self.choke_wall] + self.patrol_route)
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in protected_cells]
        np.random.shuffle(empty_cells)
        
        for i in range(num_walls):
            r, c = empty_cells.pop()
            self.grid[r, c] = self.WALL
            
        for i in range(num_penalties):
            r, c = empty_cells.pop()
            self.grid[r, c] = self.PENALTY

    def _bfs_check(self):
        def bfs(start, target, ignore_door=False):
            queue = deque([start])
            visited = set([start])
            while queue:
                r, c = queue.popleft()
                if (r, c) == target:
                    return True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if (nr, nc) not in visited:
                            cell = self.grid[nr, nc]
                            if cell != self.WALL:
                                if cell == self.DOOR and not ignore_door:
                                    continue
                                visited.add((nr, nc))
                                queue.append((nr, nc))
            return False

        path_to_key = bfs(self.start_pos, self.key_pos, ignore_door=False)
        path_to_goal = bfs(self.key_pos, self.goal_pos, ignore_door=True)
        return path_to_key and path_to_goal

    def reset(self):
        self.agent_pos = self.start_pos
        self.has_key = 0
        self.patrol_idx = 0
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return (self.agent_pos[0], self.agent_pos[1], self.has_key, self.patrol_idx)

    def step(self, action):
        self.steps += 1
        prev_pos = self.agent_pos
        
        actual_action = action
        rand_val = random.random()
        if rand_val > self.PROB_FORWARD:
            if rand_val > self.PROB_FORWARD + self.PROB_DRIFT:
                actual_action = (action + 1) % 4
            else:
                actual_action = (action - 1) % 4
                
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[actual_action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        
        hit_wall = False
        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
            cell = self.grid[nr, nc]
            if cell == self.WALL:
                hit_wall = True
                nr, nc = self.agent_pos
            elif cell == self.DOOR and self.has_key == 0:
                hit_wall = True
                nr, nc = self.agent_pos
        else:
            hit_wall = True
            nr, nc = self.agent_pos
            
        self.agent_pos = (nr, nc)
        
        self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_route)
        obstacle_pos = self.patrol_route[self.patrol_idx]
        
        key_picked_up = False
        if self.agent_pos == self.key_pos and self.has_key == 0:
            self.has_key = 1
            key_picked_up = True
            
        reward, done = self._calculate_reward_and_done(hit_wall, obstacle_pos, key_picked_up, prev_pos)
        
        return self._get_state(), reward, done, {}

    def _calculate_reward_and_done(self, hit_wall, obstacle_pos, key_picked_up, prev_pos):
        reward = -1 
        done = False
        
        if self.agent_pos == obstacle_pos:
            reward -= 50
        if hit_wall:
            reward -= 10
        if self.grid[self.agent_pos] == self.PENALTY:
            reward -= 20
        if key_picked_up:
            reward += 20 # پاداش ثابت برای دریافت کلید در نسخه Sparse
        if self.agent_pos == self.goal_pos:
            reward += 100
            done = True
            
        # نسخه دوم پاداش: اعمال Reward Shaping
        if self.use_reward_shaping and not done:
            target = self.goal_pos if self.has_key else self.key_pos
            prev_dist = abs(prev_pos[0] - target[0]) + abs(prev_pos[1] - target[1])
            curr_dist = abs(self.agent_pos[0] - target[0]) + abs(self.agent_pos[1] - target[1])
            
            # اگر به هدف نزدیک‌تر شود پاداش مثبت، اگر دور شود جریمه اضافه می‌شود
            reward += (prev_dist - curr_dist) * 2

        return reward, done

    def get_all_states(self):
        states = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] != self.WALL:
                    for k in [0, 1]:
                        for p in range(len(self.patrol_route)):
                            states.append((r, c, k, p))
        return states

    def get_transitions(self, state, action):
        r, c, k, p = state
        
        if (r, c) == self.goal_pos:
            return [(1.0, state, 0, True)]
            
        transitions = []
        action_probs = [
            (action, self.PROB_FORWARD),
            ((action + 1) % 4, self.PROB_DRIFT),
            ((action - 1) % 4, self.PROB_DRIFT)
        ]
        
        next_p = (p + 1) % len(self.patrol_route)
        obstacle_pos = self.patrol_route[next_p]
        
        for act, prob in action_probs:
            moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            dr, dc = moves[act]
            nr, nc = r + dr, c + dc
            
            hit_wall = False
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                cell = self.grid[nr, nc]
                if cell == self.WALL or (cell == self.DOOR and k == 0):
                    hit_wall = True
                    nr, nc = r, c
            else:
                hit_wall = True
                nr, nc = r, c
                
            key_picked_up = False
            next_k = k
            if (nr, nc) == self.key_pos and k == 0:
                next_k = 1
                key_picked_up = True
                
            next_state = (nr, nc, next_k, next_p)
            
            reward = -1
            done = False
            if (nr, nc) == obstacle_pos:
                reward -= 50
            if hit_wall:
                reward -= 10
            if self.grid[nr, nc] == self.PENALTY:
                reward -= 20
            if key_picked_up:
                reward += 20
            if (nr, nc) == self.goal_pos:
                reward += 100
                done = True
                
            # اعمال Reward Shaping در مدل انتقال (برای استفاده در Value Iteration)
            if self.use_reward_shaping and not done:
                target = self.goal_pos if k == 1 else self.key_pos
                prev_dist = abs(r - target[0]) + abs(c - target[1])
                curr_dist = abs(nr - target[0]) + abs(nc - target[1])
                reward += (prev_dist - curr_dist) * 2
                
            transitions.append((prob, next_state, reward, done))
            
        merged_transitions = {}
        for p_val, n_s, r_val, d_val in transitions:
            key = (n_s, r_val, d_val)
            merged_transitions[key] = merged_transitions.get(key, 0) + p_val
            
        return [(p_val, n_s, r_val, d_val) for (n_s, r_val, d_val), p_val in merged_transitions.items()]

    def generate_similar_map(self):
        new_env = copy.deepcopy(self)
        walls = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.WALL]
        num_to_move = int(len(walls) * 0.18) 
        
        removed_walls = random.sample(walls, num_to_move)
        for r, c in removed_walls:
            new_env.grid[r, c] = self.EMPTY
            
        protected = set([new_env.start_pos, new_env.key_pos, new_env.door_pos, new_env.goal_pos] + new_env.patrol_route)
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected]
        
        added_walls = random.sample(empty_cells, num_to_move)
        for r, c in added_walls:
            new_env.grid[r, c] = self.WALL
            
        if not new_env._bfs_check():
            return self.generate_similar_map()
        return new_env

    def generate_different_map(self):
        new_env = copy.deepcopy(self)
        protected = set([new_env.start_pos, new_env.door_pos, new_env.goal_pos] + new_env.patrol_route)
        
        new_env.grid[new_env.key_pos] = self.EMPTY
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected]
        new_key = random.choice(empty_cells)
        new_env.key_pos = new_key
        new_env.grid[new_key] = self.KEY
        protected.add(new_key)
        
        walls = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.WALL]
        num_to_move = int(len(walls) * 0.35)
        
        removed_walls = random.sample(walls, num_to_move)
        for r, c in removed_walls:
            new_env.grid[r, c] = self.EMPTY
            
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected]
        
        added_walls = random.sample(empty_cells, num_to_move)
        for r, c in added_walls:
            new_env.grid[r, c] = self.WALL
            
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected and (r, c) not in added_walls]
        added_penalties = random.sample(empty_cells, 3)
        for r, c in added_penalties:
            new_env.grid[r, c] = self.PENALTY
            
        if not new_env._bfs_check():
            return self.generate_different_map()
        return new_env