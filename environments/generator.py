import os
import sys
import numpy as np
from collections import deque

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.maze import DynamicMazeEnv

def validate_map_with_bfs(env):
    """اعتبارسنجی نقشه با الگوریتم BFS طبق خواسته صریح سند پروژه"""
    WALL = env.WALL
    KEY = env.KEY
    GOAL = env.GOAL
    
    def bfs(start_pos, target_type):
        queue = deque([tuple(start_pos)])
        visited = {tuple(start_pos)}
        
        while queue:
            r, c = queue.popleft()
            
            # اگر به هدف مورد نظر رسیدیم
            if env.grid[r, c] == target_type:
                return True, (r, c)
            
            # حرکت در 4 جهت اصلی
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
                    # عامل می‌تواند از همه جا عبور کند به جز دیوار
                    if env.grid[nr, nc] != WALL and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        
        return False, None

    # بررسی بخش اول: آیا مسیری از نقطه شروع به کلید وجود دارد؟
    print("Checking BFS Path: Start -> Key...")
    found_key, key_pos = bfs(env.agent_pos, KEY)
    if not found_key:
        return False
        
    # بررسی بخش دوم: آیا مسیری از کلید به هدف وجود دارد؟
    print("Checking BFS Path: Key -> Goal...")
    found_goal, _ = bfs(key_pos, GOAL)
    
    return found_goal

def generate_and_save():
    print("--- Map Generator & BFS Validator ---")
    student_id = "40411504"
    b = int(student_id[-2]) # Seed پایه
    
    os.makedirs('environments/maps', exist_ok=True)
    
    valid = False
    attempt = 0
    
    # حلقه تولید تا زمانی که یک نقشه معتبر (دارای مسیر) پیدا شود
    while not valid:
        attempt += 1
        print(f"\nAttempt {attempt} to generate valid map...")
        
        # در صورت نامعتبر بودن، سید را تغییر می‌دهیم تا نقشه جدیدی ساخته شود
        np.random.seed(b + attempt - 1)
        env = DynamicMazeEnv(use_reward_shaping=False)
        
        valid = validate_map_with_bfs(env)
        
        if valid:
            print("✅ Valid path found using BFS!")
            map_path = 'environments/maps/base_map.npy'
            np.save(map_path, env.grid)
            print(f"Map successfully saved to: {map_path}")
            break
        else:
            print("❌ Invalid map (No path found). Regenerating...")

if __name__ == "__main__":
    generate_and_save()