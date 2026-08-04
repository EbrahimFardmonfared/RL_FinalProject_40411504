import copy
from agents.q_learning import QLearningAgent

class TransferLearningExperiment:
    def __init__(self, source_env, target_env):
        self.source_env = source_env
        self.target_env = target_env
        self.source_q_table = None

    def train_source(self, episodes=500):
        agent = QLearningAgent(self.source_env)
        # رفع باگ: استخراج تاپل ۴ تایی به جای دیکشنری
        rewards, steps, wall_hits, penalty_hits = agent.train(episodes=episodes)
        
        # ذخیره Q-table مبدأ برای انتقال
        self.source_q_table = copy.deepcopy(agent.Q)
        return rewards, steps, wall_hits, penalty_hits

    def train_target_from_scratch(self, episodes=500):
        agent = QLearningAgent(self.target_env)
        rewards, steps, wall_hits, penalty_hits = agent.train(episodes=episodes)
        return rewards, steps, wall_hits, penalty_hits

    def train_target_with_transfer(self, episodes=500):
        agent = QLearningAgent(self.target_env)
        
        # انتقال جدول Q از محیط مبدأ
        if self.source_q_table is not None:
            agent.Q = copy.deepcopy(self.source_q_table)
            
        # کاهش مقدار اکتشاف چون عامل از قبل دانش اولیه‌ای دارد
        agent.epsilon = 0.5 
        
        rewards, steps, wall_hits, penalty_hits = agent.train(episodes=episodes)
        return rewards, steps, wall_hits, penalty_hits