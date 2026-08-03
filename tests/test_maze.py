import sys
import os
import unittest

# اضافه کردن مسیر ریشه پروژه برای دسترسی به ماژول‌ها
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent

class TestDynamicMaze(unittest.TestCase):
    
    def setUp(self):
        """راه‌اندازی محیط پیش از اجرای هر تست"""
        self.env = DynamicMazeEnv(use_reward_shaping=False)
        
    def test_environment_initialization(self):
        """تست ابعاد نقشه و مقداردهی اولیه عامل"""
        self.assertEqual(self.env.grid_size, 15, "Grid size must be 15x15 based on the seed.")
        state = self.env.reset()
        self.assertEqual(len(state), 4, "State representation must have 4 dimensions (r, c, key, patrol).")
        self.assertEqual(state[0:2], self.env.start_pos, "Agent must start at start_pos.")
        self.assertEqual(state[2], 0, "Agent must not have the key initially.")
        
    def test_map_elements_distribution(self):
        """تست توزیع صحیح موانع و خانه‌های جریمه (حداقل 15% مانع و 5 جریمه)"""
        wall_count = sum([1 for r in range(self.env.grid_size) for c in range(self.env.grid_size) if self.env.grid[r, c] == self.env.WALL])
        penalty_count = sum([1 for r in range(self.env.grid_size) for c in range(self.env.grid_size) if self.env.grid[r, c] == self.env.PENALTY])
        
        self.assertGreaterEqual(wall_count, 33, "Map must contain at least 15% walls.")
        self.assertGreaterEqual(penalty_count, 5, "Map must contain at least 5 penalty cells.")
        
    def test_bfs_validity(self):
        """تست اطمینان از حل‌پذیر بودن نقشه (وجود مسیر معتبر به کلید و هدف)"""
        # تابع _bfs_check مقدار True برمی‌گرداند اگر نقشه قابل حل باشد
        is_solvable = self.env._bfs_check()
        self.assertTrue(is_solvable, "The generated map must have a valid path to the key and goal.")

    def test_q_learning_agent_update(self):
        """تست منطق به‌روزرسانی جدول Q در الگوریتم Q-Learning"""
        agent = QLearningAgent(self.env)
        state = self.env.reset()
        action = 1 # فرض می‌کنیم عمل 'راست' را انتخاب می‌کند
        
        initial_q = agent.get_q(state, action)
        self.assertEqual(initial_q, 0.0, "Initial Q-value must be zero.")
        
        # انجام یک به‌روزرسانی دستی برای تست فرمول
        reward = -1
        next_state = (0, 1, 0, 1)
        best_next_q = 0.0
        
        agent.Q[(state, action)] = initial_q + agent.alpha * (reward + agent.gamma * best_next_q - initial_q)
        updated_q = agent.get_q(state, action)
        
        self.assertEqual(updated_q, -0.1, "Q-value was not updated correctly based on the Bellman equation.")

if __name__ == '__main__':
    unittest.main()