import random

class SarsaLambdaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, lmbda=0.9, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.995, trace_type='replacing'):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda  # پارامتر لامبدا برای تنظیم عمق ردپای شایستگی
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.trace_type = trace_type
        
        self.Q = {}
        self.E = {} # Eligibility Traces (ردپای شایستگی)
        
        self.logs = {
            'rewards': [],
            'steps': [],
            'success': [],
            'wall_hits': [],
            'penalty_hits': []
        }

    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)
        
    def get_e(self, state, action):
        return self.E.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            q_values = [self.get_q(state, a) for a in range(4)]
            max_q = max(q_values)
            best_actions = [a for a in range(4) if q_values[a] == max_q]
            return random.choice(best_actions)

    def train(self, episodes=500, max_steps=500):
        for ep in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            
            # در شروع هر اپیزود، تمام ردپاها پاک می‌شوند
            self.E = {}
            
            total_reward = 0
            steps = 0
            wall_hits = 0
            penalty_hits = 0
            success = 0
            
            for step in range(max_steps):
                next_state, reward, done, _ = self.env.step(action)
                
                # انتخاب عمل بعدی همین الان انجام می‌شود (On-policy)
                next_action = self.choose_action(next_state)
                
                if reward == -10:
                    wall_hits += 1
                elif reward == -20:
                    penalty_hits += 1
                elif reward == 100:
                    success = 1
                    
                # محاسبه خطای TD
                current_q = self.get_q(state, action)
                next_q = self.get_q(next_state, next_action)
                
                if done:
                    delta = reward - current_q
                else:
                    delta = reward + self.gamma * next_q - current_q
                    
                # بروزرسانی ردپای شایستگی برای حالت فعلی
                if self.trace_type == 'replacing':
                    self.E[(state, action)] = 1.0
                else: # Accumulating
                    self.E[(state, action)] = self.get_e(state, action) + 1.0
                    
                # بروزرسانی Q و E برای تمام جفت‌های (حالت-عمل) ویزیت شده
                for (s, a), e_val in list(self.E.items()):
                    self.Q[(s, a)] = self.get_q(s, a) + self.alpha * delta * e_val
                    # کاهش تدریجی ردپا
                    self.E[(s, a)] = self.gamma * self.lmbda * e_val
                    
                    # بهینه‌سازی حافظه: حذف ردپاهای بسیار کوچک
                    if self.E[(s, a)] < 1e-4:
                        del self.E[(s, a)]
                        
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
                
                if done:
                    break
                    
            self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)
            
            self.logs['rewards'].append(total_reward)
            self.logs['steps'].append(steps)
            self.logs['success'].append(success)
            self.logs['wall_hits'].append(wall_hits)
            self.logs['penalty_hits'].append(penalty_hits)

        print(f"SARSA(lambda={self.lmbda}) completed {episodes} episodes.")
        return self.logs