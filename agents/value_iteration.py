import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        self.V = {}
        self.policy = {}
        
        # مقداردهی اولیه به صفر برای تمام حالت‌های معتبر
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if env.grid[r, c] == env.WALL:
                    continue
                for has_key in [0, 1]:
                    for patrol_idx in range(len(env.patrol_route)):
                        self.V[(r, c, has_key, patrol_idx)] = 0.0

    def run(self):
        """اجرای الگوریتم Value Iteration با استفاده از مدل احتمالاتی واقعی انتقال"""
        iterations = 0
        while True:
            delta = 0
            new_V = self.V.copy()
            
            for state in self.V.keys():
                action_values = []
                for action in range(4):
                    # دریافت توزیع احتمال (احتمال، حالت_بعدی، پاداش، پایان) از محیط
                    transitions = self.env.get_transitions(state, action)
                    expected_val = 0.0
                    
                    for prob, next_state, reward, done in transitions:
                        if done:
                            expected_val += prob * reward
                        else:
                            expected_val += prob * (reward + self.gamma * self.V[next_state])
                            
                    action_values.append(expected_val)
                
                # آپدیت مقدار حالت
                best_value = max(action_values)
                new_V[state] = best_value
                delta = max(delta, abs(best_value - self.V[state]))
            
            self.V = new_V
            iterations += 1
            
            # شرط توقف (همگرایی)
            if delta < self.theta:
                break
                
        self._extract_policy()
        return iterations

    def _extract_policy(self):
        """استخراج سیاست بهینه (Policy) بر اساس مقادیر V همگرا شده"""
        for state in self.V.keys():
            action_values = []
            for action in range(4):
                transitions = self.env.get_transitions(state, action)
                expected_val = 0.0
                for prob, next_state, reward, done in transitions:
                    if done:
                        expected_val += prob * reward
                    else:
                        expected_val += prob * (reward + self.gamma * self.V[next_state])
                action_values.append(expected_val)
                
            self.policy[state] = int(np.argmax(action_values))

    def get_action(self, state):
        """بازگرداندن اکشن بر اساس سیاست پیدا شده"""
        return self.policy.get(state, 0)