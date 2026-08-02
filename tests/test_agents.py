import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.env = DynamicMazeEnv()
        self.agent = QLearningAgent(self.env, alpha=0.1, gamma=0.99)

    def test_q_table_initialization(self):
        """تست مقداردهی اولیه و صحیح جدول Q"""
        self.assertIsInstance(self.agent.Q, dict)
        state = self.env.reset()
        # مقدار اولیه برای حالتی که دیده نشده باید صفر باشد
        self.assertEqual(self.agent.get_q(state, 0), 0.0)

    def test_action_selection(self):
        """تست انتخاب عمل معتبر توسط سیاست عامل"""
        state = self.env.reset()
        action = self.agent.choose_action(state)
        self.assertIn(action, [0, 1, 2, 3], "Action must be in range 0-3 (UP, DOWN, LEFT, RIGHT)")

if __name__ == '__main__':
    unittest.main()