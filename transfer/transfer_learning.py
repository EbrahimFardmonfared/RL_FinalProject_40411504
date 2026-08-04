import copy
from agents.q_learning import QLearningAgent

class TransferLearningExperiment:
    def __init__(self, source_env, target_env):
        self.source_env = source_env
        self.target_env = target_env
        self.source_q_table = None

    def train_source(self, episodes=500, max_steps=500):
        agent = QLearningAgent(self.source_env)
        rewards, steps, wall_hits, penalty_hits = agent.train(episodes=episodes, max_steps=max_steps)
        self.source_q_table = copy.deepcopy(agent.Q)
        return rewards, steps, wall_hits, penalty_hits

    def train_target_from_scratch(self, episodes=500, max_steps=500):
        agent = QLearningAgent(self.target_env)
        return agent.train(episodes=episodes, max_steps=max_steps)

    def train_target_full_transfer(self, episodes=500, max_steps=500):
        agent = QLearningAgent(self.target_env)
        if self.source_q_table is not None:
            agent.Q = copy.deepcopy(self.source_q_table)
        agent.epsilon = 0.5  # کاهش اکتشاف به‌خاطر دانش قبلی
        return agent.train(episodes=episodes, max_steps=max_steps)

    def train_target_beta_transfer(self, episodes=500, max_steps=500, beta=0.5):
        agent = QLearningAgent(self.target_env)
        if self.source_q_table is not None:
            # اعمال ضریب بتا روی دانش مبدأ
            for key, val in self.source_q_table.items():
                agent.Q[key] = beta * val
        agent.epsilon = 0.5
        return agent.train(episodes=episodes, max_steps=max_steps)

    def train_target_selective_transfer(self, episodes=500, max_steps=500):
        agent = QLearningAgent(self.target_env)
        if self.source_q_table is not None:
            for (state, action), q_val in self.source_q_table.items():
                r, c = state[0], state[1]
                # انتقال انتخابی: فقط به خانه‌هایی که در مقصد دیوار نیستند منتقل کن
                if 0 <= r < self.target_env.grid_size and 0 <= c < self.target_env.grid_size:
                    if self.target_env.grid[r, c] != self.target_env.WALL:
                        agent.Q[(state, action)] = q_val
        agent.epsilon = 0.5
        return agent.train(episodes=episodes, max_steps=max_steps)