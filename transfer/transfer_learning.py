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
                
                # 🔴 پیاده‌سازی همسایگی محلی (Local Neighborhood): خود خانه و 4 خانه اطراف
                neighborhood_match = True
                for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.source_env.grid_size and 0 <= nc < self.source_env.grid_size:
                        # اگر ساختار این خانه در مبدأ و مقصد فرق داشت، همسایگی به هم خورده است
                        if self.source_env.grid[nr, nc] != self.target_env.grid[nr, nc]:
                            neighborhood_match = False
                            break
                
                # انتقال انتخابی: فقط اگر کل همسایگی سالم بود، دانش را منتقل کن
                if neighborhood_match:
                    agent.Q[(state, action)] = q_val
                    
        agent.epsilon = 0.5
        return agent.train(episodes=episodes, max_steps=max_steps)