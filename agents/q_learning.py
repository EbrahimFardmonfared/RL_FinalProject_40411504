import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.995, decay_type='exponential'):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        self.Q = defaultdict(float)

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, 3) 
        else:
            q_values = [self.get_q(state, a) for a in range(4)]
            max_q = max(q_values)
            best_actions = [a for a in range(4) if q_values[a] == max_q]
            return random.choice(best_actions)

    def train(self, episodes=1000, max_steps=500):
        rewards = []
        steps = []
        wall_hits_list = []
        penalty_hits_list = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            step = 0
            wall_hits = 0
            penalty_hits = 0
            done = False
            
            # اضافه‌شدن شرط max_steps برای جلوگیری از حلقه‌ی بی‌نهایت
            while not done and step < max_steps:
                action = self.choose_action(state, self.epsilon)
                next_state, reward, done, info = self.env.step(action)
                
                best_next_action = np.argmax([self.get_q(next_state, a) for a in range(4)])
                td_target = reward + self.gamma * self.get_q(next_state, best_next_action)
                td_error = td_target - self.get_q(state, action)
                self.Q[(state, action)] = self.get_q(state, action) + self.alpha * td_error
                
                if info.get('hit_wall', False):
                    wall_hits += 1
                if info.get('hit_penalty', False):
                    penalty_hits += 1
                    
                state = next_state
                total_reward += reward
                step += 1
                
            if self.decay_type == 'exponential':
                self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)
            elif self.decay_type == 'linear':
                self.epsilon = max(self.epsilon_end, self.epsilon - self.decay_rate)
                
            rewards.append(total_reward)
            steps.append(step)
            wall_hits_list.append(wall_hits)
            penalty_hits_list.append(penalty_hits)
            
        return rewards, steps, wall_hits_list, penalty_hits_list