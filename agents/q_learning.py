import random
import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.995, decay_type='exponential'):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_rate = decay_rate
        self.decay_type = decay_type
        
        # جدول Q به صورت دیکشنری (برای مدیریت راحت‌تر فضای حالت)
        self.Q = {}
        
        # دیکشنری لاگ‌ها برای رسم نمودار و تحلیل در گزارش
        self.logs = {
            'rewards': [],
            'steps': [],
            'success': [],
            'wall_hits': [],
            'penalty_hits': []
        }

    def get_q(self, state, action):
        """بازگرداندن مقدار Q؛ اگر قبلاً دیده نشده باشد، مقدار اولیه صفر است"""
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state):
        """انتخاب عمل بر اساس سیاست حریصانه (Epsilon-Greedy)"""
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        else:
            q_values = [self.get_q(state, a) for a in range(4)]
            max_q = max(q_values)
            # در صورت برابر بودن مقادیر، یکی از بهترین‌ها تصادفی انتخاب می‌شود
            best_actions = [a for a in range(4) if q_values[a] == max_q]
            return random.choice(best_actions)

    def decay_epsilon(self, current_episode, total_episodes):
        """کاهش مقدار اپسیلون بر اساس دو روش خواسته شده"""
        if self.decay_type == 'exponential':
            self.epsilon = max(self.epsilon_end, self.epsilon * self.decay_rate)
        elif self.decay_type == 'linear':
            drop_per_episode = (self.epsilon_start - self.epsilon_end) / total_episodes
            self.epsilon = max(self.epsilon_end, self.epsilon - drop_per_episode)

    def train(self, episodes=1000, max_steps=500):
        """اجرای حلقه اصلی آموزش الگوریتم"""
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            wall_hits = 0
            penalty_hits = 0
            success = 0
            
            for step in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # ثبت رویدادها برای لاگ‌گیری (طبق مقادیر پاداش در محیط)
                if reward == -10:
                    wall_hits += 1
                elif reward == -20:
                    penalty_hits += 1
                elif reward == 100:
                    success = 1
                    
                # فرمول بروزرسانی Q-Learning (Off-policy)
                best_next_q = max([self.get_q(next_state, a) for a in range(4)])
                current_q = self.get_q(state, action)
                self.Q[(state, action)] = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
                    
            # کاهش اپسیلون در انتهای هر اپیزود
            self.decay_epsilon(ep, episodes)
            
            # ذخیره لاگ‌های این اپیزود
            self.logs['rewards'].append(total_reward)
            self.logs['steps'].append(steps)
            self.logs['success'].append(success)
            self.logs['wall_hits'].append(wall_hits)
            self.logs['penalty_hits'].append(penalty_hits)

        print(f"Q-Learning ({self.decay_type} decay) completed {episodes} episodes.")
        return self.logs