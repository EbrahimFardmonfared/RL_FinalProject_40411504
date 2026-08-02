import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.maze import DynamicMazeEnv

class TestDynamicMaze(unittest.TestCase):
    def setUp(self):
        self.env = DynamicMazeEnv()

    def test_grid_initialization(self):
        """تست ساخته شدن درست ابعاد شبکه و وجود عناصر کلیدی"""
        self.assertIsNotNone(self.env.grid)
        self.assertEqual(self.env.grid.shape[0], self.env.grid_size)
        
        # بررسی وجود نقطه شروع، هدف و کلید در نقشه
        unique_elements = set(self.env.grid.flatten())
        self.assertIn(self.env.START, unique_elements)
        self.assertIn(self.env.GOAL, unique_elements)
        self.assertIn(self.env.KEY, unique_elements)

    def test_state_space_markov_property(self):
        """تست حفظ خاصیت مارکف در نمایش حالت (x, y, has_key)"""
        state = self.env.reset()
        self.assertEqual(len(state), 3, "State must have 3 dimensions")
        self.assertIn(state[2], [0, 1], "Key status must be boolean (0 or 1)")

if __name__ == '__main__':
    unittest.main()