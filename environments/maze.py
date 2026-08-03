import numpy as np
import random
import copy
from collections import deque

class DynamicMazeEnv:
    def __init__(self, use_reward_shaping=False):
        self.use_reward_shaping = use_reward_shaping
        # تنظیمات پایه بر اساس شماره دانشجویی 40411504
        self.seed = 0
        self.grid_size = 15
        
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
        
        # مسیر حلقه‌ای مانع متحرک (مختصات فرضی که در نقشه خالی خواهند بود)
        self.patrol_route = [(7, 7), (7, 8), (7, 9), (8, 9), (9, 9), (9, 8), (9, 7), (8, 7)]
         
        # تولید و اعتبارسنجی نقشه
        self.grid = None
        self._generate_valid_map()
        
        # وضعیت اولیه عامل
        self.reset()

    def _generate_valid_map(self):
        """تولید نقشه تا زمانی که یک نقشه معتبر (حل‌پذیر) پیدا شود."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        is_valid = False
        while not is_valid:
            self._build_grid()
            is_valid = self._bfs_check()
            if not is_valid:
                # تغییر موقت سید برای تلاش مجدد در صورت بسته بودن مسیر
                self.seed += 1 
                np.random.seed(self.seed)
                random.seed(self.seed)

    def _build_grid(self):
        """ساختاردهی اولیه نقشه، دیوارها، جریمه‌ها و المان‌های کلیدی"""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # جایگذاری المان‌های اصلی با ایجاد گلوگاه اجباری برای در
        self.start_pos = (0, 0)
        self.key_pos = (2, 12)
        self.goal_pos = (14, 14)
        self.door_pos = (13, 14) # در دقیقاً چسبیده به هدف قرار می‌گیرد
        self.choke_wall = (14, 13) # مسدود کردن تنها راه فرعی به هدف
        
        self.grid[self.start_pos] = self.START
        self.grid[self.key_pos] = self.KEY
        self.grid[self.goal_pos] = self.GOAL
        self.grid[self.door_pos] = self.DOOR
        self.grid[self.choke_wall] = self.WALL # این دیوار تضمین می‌کند که در دور زده نشود
        
        # اضافه کردن حداقل 15 درصد مانع (15 * 15 * 0.15 ≈ 34 دیوار)
        num_walls = 34
        # اضافه کردن حداقل 5 خانه جریمه
        num_penalties = 6
        
        # محافظت از مسیرها تا دیوار روی آن‌ها قرار نگیرد
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
        """بررسی وجود مسیر از شروع به کلید و از کلید به هدف"""
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
                            # دیوارها همیشه غیرقابل عبورند. در بسته فقط وقتی نادیده گرفته می‌شود که کلید را داشته باشیم.
                            if cell != self.WALL:
                                if cell == self.DOOR and not ignore_door:
                                    continue
                                visited.add((nr, nc))
                                queue.append((nr, nc))
            return False

        # 1. مسیری از شروع به کلید وجود داشته باشد (بدون عبور از در بسته)
        path_to_key = bfs(self.start_pos, self.key_pos, ignore_door=False)
        # 2. مسیری از کلید به هدف وجود داشته باشد (در این مرحله در بسته قابل عبور است)
        path_to_goal = bfs(self.key_pos, self.goal_pos, ignore_door=True)
        
        return path_to_key and path_to_goal

    def reset(self):
        """بازگرداندن محیط به حالت اولیه در شروع هر اپیزود"""
        self.agent_pos = self.start_pos
        self.has_key = 0
        self.patrol_idx = 0
        self.steps = 0
        # تعریف حالت بر اساس فرمول: (x, y, k, p)
        return self._get_state()

    def _get_state(self):
        return (self.agent_pos[0], self.agent_pos[1], self.has_key, self.patrol_idx)

    def step(self, action):
        r, c = self.agent_pos
        
        # 1. پیاده‌سازی عدم قطعیت محیط (Drift 0.8 / 0.1 / 0.1)
        import random
        rand_val = random.random()
        if rand_val < 0.1:     # 10% انحراف به راست
            action = (action + 1) % 4
        elif rand_val < 0.2:   # 10% انحراف به چپ
            action = (action - 1) % 4
            
        # 2. محاسبه مختصات جدید
        nr, nc = r, c
        if action == 0:   nr -= 1  # Up
        elif action == 1: nc += 1  # Right
        elif action == 2: nr += 1  # Down
        elif action == 3: nc -= 1  # Left
        
        # 3. بررسی برخوردها (دیوار، مرزها و در قفل)
        hit_wall = False
        hit_door_locked = False
        
        if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
            hit_wall = True
            nr, nc = r, c  # خنثی شدن حرکت
        elif self.grid[nr, nc] == self.WALL:
            hit_wall = True
            nr, nc = r, c
        elif self.grid[nr, nc] == self.DOOR and not self.has_key:
            hit_door_locked = True
            nr, nc = r, c
            
        self.agent_pos = [nr, nc]
        
        # 4. آپدیت وضعیت مانع متحرک
        self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_route)
        obs_r, obs_c = self.patrol_route[self.patrol_idx]
        
        # 5. بررسی وضعیت خانه جدید و برخورد با مانع
        cell_type = self.grid[nr, nc]
        hit_penalty = (cell_type == self.PENALTY)
        hit_obstacle = (self.agent_pos[0] == obs_r and self.agent_pos[1] == obs_c)
        
        done = (cell_type == self.GOAL)
        
        # 6. محاسبه پاداش پایه و پر کردن دیکشنری info
        reward = -1  # هزینه هر گام
        info = {
            'hit_wall': False, 'hit_penalty': False, 'got_key': False, 
            'hit_obstacle': False, 'door_locked_bump': False, 
            'door_passed': False, 'timeout': False
        }

        if hit_wall:
            reward = -10
            info['hit_wall'] = True
        elif hit_penalty:
            reward = -20
            info['hit_penalty'] = True
        elif hit_obstacle:
            reward = -50
            info['hit_obstacle'] = True
        elif hit_door_locked:
            info['door_locked_bump'] = True
        elif cell_type == self.GOAL:
            reward = 100
        elif cell_type == self.KEY and not self.has_key:
            reward = 20  # پاداش کلید برگشت!
            self.has_key = 1
            info['got_key'] = True
        elif cell_type == self.DOOR and self.has_key:
            info['door_passed'] = True

        # === 7. اعمال Reward Shaping ===
        if self.use_reward_shaping and not done:
            # تابع پتانسیل: فاصله منهتن تا هدف (قرینه)
            phi_current = - (abs(r - 14) + abs(c - 7)) # مختصات هدف (14,7) است
            phi_next = - (abs(self.agent_pos[0] - 14) + abs(self.agent_pos[1] - 7))
            gamma = 0.9 # ضریب تخفیف
            shaping_reward = (gamma * phi_next) - phi_current
            reward += shaping_reward

        # 8. ساختن حالت بعدی
        state = (self.agent_pos[0], self.agent_pos[1], self.has_key, self.patrol_idx)
        return state, reward, done, info

    def _calculate_reward_and_done(self, hit_wall, obstacle_pos):
        reward = -1 # هزینه هر حرکت (Step Penalty)
        done = False
        
        # برخورد با مانع متحرک
        if self.agent_pos == obstacle_pos:
            reward = -50
            
        # برخورد با دیوار
        if hit_wall:
            reward = -10
            
        # ورود به خانه جریمه
        if self.grid[self.agent_pos] == self.PENALTY:
            reward = -20
            
        # رسیدن به هدف
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
            
        # در فازهای بعدی Reward Shaping را می‌توانیم در همین تابع پیاده کنیم
        return reward, done

    def get_all_states(self):
        """تولید تمام حالت‌های معتبر برای مقداردهی اولیه V(s)"""
        states = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] != self.WALL:
                    for k in [0, 1]:
                        for p in range(len(self.patrol_route)):
                            states.append((r, c, k, p))
        return states

    def get_transitions(self, state, action):
        """
        استخراج مدل انتقال P(s'|s,a) و پاداش برای معادله بلمن
        خروجی: لیستی از تاپل‌ها به فرم (probability, next_state, reward, done)
        """
        r, c, k, p = state
        
        # اگر عامل در خانه هدف است، اپیزود تمام شده و انتقالی نداریم
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
                
            next_k = 1 if (nr, nc) == self.key_pos else k
            next_state = (nr, nc, next_k, next_p)
            
            # استفاده از همان منطق پاداش
            reward = -1
            done = False
            if (nr, nc) == obstacle_pos:
                reward = -50
            if hit_wall:
                reward = -10
            if self.grid[nr, nc] == self.PENALTY:
                reward = -20
            if (nr, nc) == self.goal_pos:
                reward = 100
                done = True
                
            transitions.append((prob, next_state, reward, done))
            
        # ترکیب احتمالات تکراری در صورت برخورد به دیوار
        merged_transitions = {}
        for p_val, n_s, r_val, d_val in transitions:
            key = (n_s, r_val, d_val)
            merged_transitions[key] = merged_transitions.get(key, 0) + p_val
            
        return [(p_val, n_s, r_val, d_val) for (n_s, r_val, d_val), p_val in merged_transitions.items()]

    def generate_similar_map(self):
        """ساخت محیط مقصد مشابه: جابجایی حدود 18 درصد موانع با حفظ سایر المان‌ها"""
        new_env = copy.deepcopy(self)
        
        # استخراج مختصات تمام دیوارهای فعلی
        walls = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.WALL]
        num_to_move = int(len(walls) * 0.18) # جابجایی حدود 18 درصد (بین 15 تا 20 درصد خواسته شده)
        
        # حذف تصادفی تعدادی از دیوارها
        removed_walls = random.sample(walls, num_to_move)
        for r, c in removed_walls:
            new_env.grid[r, c] = self.EMPTY
            
        # پیدا کردن خانه‌های خالی مجاز برای قرار دادن دیوارهای جدید
        protected = set([new_env.start_pos, new_env.key_pos, new_env.door_pos, new_env.goal_pos] + new_env.patrol_route)
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected]
        
        # اضافه کردن دیوارهای جدید
        added_walls = random.sample(empty_cells, num_to_move)
        for r, c in added_walls:
            new_env.grid[r, c] = self.WALL
            
        # بررسی حل‌پذیری با BFS؛ اگر نقشه بسته شده بود، دوباره تلاش کن
        if not new_env._bfs_check():
            return self.generate_similar_map()
            
        return new_env

    def generate_different_map(self):
        """ساخت محیط مقصد متفاوت: جابجایی 35 درصد موانع، تغییر مکان کلید، افزودن 3 جریمه جدید"""
        new_env = copy.deepcopy(self)
        protected = set([new_env.start_pos, new_env.door_pos, new_env.goal_pos] + new_env.patrol_route)
        
        # 1. تغییر مکان کلید
        new_env.grid[new_env.key_pos] = self.EMPTY
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected]
        new_key = random.choice(empty_cells)
        new_env.key_pos = new_key
        new_env.grid[new_key] = self.KEY
        protected.add(new_key)
        
        # 2. جابجایی 35 درصد موانع
        walls = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if new_env.grid[r, c] == self.WALL]
        num_to_move = int(len(walls) * 0.35)
        
        removed_walls = random.sample(walls, num_to_move)
        for r, c in removed_walls:
            new_env.grid[r, c] = self.EMPTY
            
        # آپدیت لیست خانه‌های خالی بعد از حذف دیوارها
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected]
        
        added_walls = random.sample(empty_cells, num_to_move)
        for r, c in added_walls:
            new_env.grid[r, c] = self.WALL
            
        # 3. افزودن 3 خانه جریمه جدید
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) 
                       if new_env.grid[r, c] == self.EMPTY and (r, c) not in protected and (r, c) not in added_walls]
        added_penalties = random.sample(empty_cells, 3)
        for r, c in added_penalties:
            new_env.grid[r, c] = self.PENALTY
            
        # بررسی حل‌پذیری
        if not new_env._bfs_check():
            return self.generate_different_map()
            
        return new_env