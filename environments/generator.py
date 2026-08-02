import os
import sys
import numpy as np

# اضافه کردن مسیر ریشه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.maze import DynamicMazeEnv

def generate_and_save_map(student_id="40411504"):
    """تولید نقشه پایه بر اساس فرمول شماره دانشجویی و ذخیره آن"""
    # استخراج رقم یکی مانده به آخر طبق صورت پروژه
    b = int(student_id[-2])
    N = 15 + (b % 4)
    
    print(f"Generating map for Student ID: {student_id}")
    print(f"Base Seed: {b} | Maze Size: {N}x{N}")
    
    # تنظیم Seed برای تکرارپذیری
    np.random.seed(b)
    
    # مقداردهی محیط
    env = DynamicMazeEnv()
    
    # مسیر ذخیره‌سازی
    save_dir = "environments/maps"
    os.makedirs(save_dir, exist_ok=True)
    map_path = os.path.join(save_dir, "base_map.npy")
    
    # ذخیره ماتریس نقشه
    np.save(map_path, env.grid)
    print(f"Map successfully saved to {map_path}")

if __name__ == "__main__":
    generate_and_save_map()