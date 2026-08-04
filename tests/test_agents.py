import unittest
import sys
import os

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.env = DynamicMazeEnv()
        self.q_agent = QLearningAgent(self.env)
        self.sarsa_agent = SarsaLambdaAgent(self.env)
        # فرمت وضعیت: (ردیف، ستون، کلید دارد/ندارد، ایندکس مانع)
        self.state = (0, 0, 0, 0)

    def test_action_selection(self):
        # رفع باگ: ارسال آرگومان epsilon برای متد choose_action که اجباری شده است
        action_q = self.q_agent.choose_action(self.state, epsilon=0.0)
        self.assertIn(action_q, [0, 1, 2, 3], "Q-Learning action should be valid (0-3)")
        
        action_sarsa = self.sarsa_agent.choose_action(self.state, epsilon=0.0)
        self.assertIn(action_sarsa, [0, 1, 2, 3], "SARSA action should be valid (0-3)")

    def test_q_table_update(self):
        # تست آپدیت ساده جدول Q
        initial_q = self.q_agent.get_q(self.state, 0)
        self.q_agent.Q[(self.state, 0)] = 15.5
        updated_q = self.q_agent.get_q(self.state, 0)
        
        self.assertNotEqual(initial_q, updated_q)
        self.assertEqual(updated_q, 15.5)

if __name__ == '__main__':
    unittest.main()