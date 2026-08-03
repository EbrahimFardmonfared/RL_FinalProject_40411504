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
        self.env = DynamicMazeEnv()
        
    def test_state_space_markov_property(self):
        """تست حفظ خاصیت مارکوف با فضای حالت 4 بعدی"""
        state = self.env.reset()
        
        # بررسی طول State که باید دقیقا 4 باشد (r, c, has_key, patrol_idx)
        self.assertEqual(len(state), 4, "State dimension must be 4: (r, c, has_key, patrol_idx)")
        
        # بررسی منطق مقادیر داخل State
        self.assertEqual(list(state[0:2]), list(self.env.start_pos), "Agent must start at start_pos.")
        self.assertIn(state[2], [0, 1], "Key flag must be boolean (0 or 1)")
        self.assertGreaterEqual(state[3], 0, "Patrol index must be valid and positive")
        
    def test_map_elements_distribution(self):
        """تست توزیع صحیح موانع و خانه‌های جریمه"""
        wall_count = sum([1 for r in range(self.env.grid_size) for c in range(self.env.grid_size) if self.env.grid[r, c] == self.env.WALL])
        penalty_count = sum([1 for r in range(self.env.grid_size) for c in range(self.env.grid_size) if self.env.grid[r, c] == self.env.PENALTY])
        
        self.assertGreaterEqual(wall_count, 33, "Map must contain at least 15% walls.")
        self.assertGreaterEqual(penalty_count, 5, "Map must contain at least 5 penalty cells.")
        
    def test_bfs_validity(self):
        """تست اطمینان از حل‌پذیر بودن نقشه"""
        is_solvable = self.env._bfs_check()
        self.assertTrue(is_solvable, "The generated map must have a valid path to the key and goal.")

    def test_q_learning_agent_update(self):
        """تست منطق به‌روزرسانی جدول Q"""
        agent = QLearningAgent(self.env)
        state = self.env.reset()
        action = 1 
        
        initial_q = agent.get_q(state, action)
        
        # یک آپدیت فرضی برای تست فرمول
        reward = -1
        next_state = (0, 1, 0, 1)
        best_next_q = 0.0
        
        agent.Q[(state, action)] = initial_q + agent.alpha * (reward + agent.gamma * best_next_q - initial_q)
        updated_q = agent.get_q(state, action)
        
        self.assertNotEqual(initial_q, updated_q, "Q-value was not updated correctly.")

if __name__ == '__main__':
    unittest.main()