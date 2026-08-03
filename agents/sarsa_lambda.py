import numpy as np
import random
from collections import defaultdict

class SarsaLambdaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, lmbda=0.9, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.995, trace_type='replacing'):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.trace_type = trace_type
        
        self.Q = defaultdict(float)
        self.E = defaultdict(float) 

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            # رفع باگ action_space
            return random.randint(0, 3)
        else:
            # رفع باگ action_space
            q_values = [self.get_q(state, a) for a in range(4)]
            max_q = max(q_values)
            best_actions = [a for a in range(4) if q_values[a] == max_q]
            return random.choice(best_actions)

    def train(self, episodes=1000):
        rewards = []
        steps = []
        wall_hits_list = []
        penalty_hits_list = []
        
        for episode in range(episodes):
            self.E.clear() 
            
            state = self.env.reset()
            action = self.choose_action(state, self.epsilon)
            
            total_reward = 0
            step = 0
            wall_hits = 0
            penalty_hits = 0
            done = False
            
            while not done:
                next_state, reward, done, info = self.env.step(action)
                next_action = self.choose_action(next_state, self.epsilon)
                
                td_target = reward + self.gamma * self.get_q(next_state, next_action)
                td_error = td_target - self.get_q(state, action)
                
                if self.trace_type == 'accumulating':
                    self.E[(state, action)] = self.E.get((state, action), 0.0) + 1.0
                else: 
                    self.E[(state, action)] = 1.0
                    
                for (s, a) in list(self.E.keys()):
                    self.Q[(s, a)] = self.get_q(s, a) + self.alpha * td_error * self.E[(s, a)]
                    self.E[(s, a)] *= self.gamma * self.lmbda
                    
                    if self.E[(s, a)] < 1e-4:
                        del self.E[(s, a)]
                        
                if info.get('hit_wall', False):
                    wall_hits += 1
                if info.get('hit_penalty', False):
                    penalty_hits += 1
                    
                state = next_state
                action = next_action
                total_reward += reward
                step += 1
                
            self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)
            
            rewards.append(total_reward)
            steps.append(step)
            wall_hits_list.append(wall_hits)
            penalty_hits_list.append(penalty_hits)
            
        return rewards, steps, wall_hits_list, penalty_hits_list