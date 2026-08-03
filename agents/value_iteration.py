import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # V-table: فضای حالت ۴ بعدی (ردیف، ستون، داشتن کلید، ایندکس مانع)
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

    def _get_transition_prob_reward(self, state, action):
        """شبیه‌سازی یک قدم در محیط برای دریافت حالت بعدی و پاداش"""
        # ذخیره وضعیت فعلی محیط برای بازگرداندن آن بعد از شبیه‌سازی
        original_state = self.env.agent_pos, self.env.has_key, self.env.patrol_idx
        
        # تنظیم محیط به حالت مورد نظر
        self.env.agent_pos = [state[0], state[1]]
        self.env.has_key = bool(state[2])
        self.env.patrol_idx = state[3]
        
        # اجرای اکشن
        next_state, reward, done, _ = self.env.step(action)
        
        # بازگرداندن محیط به وضعیت اولیه
        self.env.agent_pos = original_state[0]
        self.env.has_key = original_state[1]
        self.env.patrol_idx = original_state[2]
        
        return next_state, reward, done

    def run(self):
        """اجرای الگوریتم Value Iteration و محاسبه V-table"""
        iterations = 0
        while True:
            delta = 0
            # کپی از مقادیر قبلی
            new_V = self.V.copy()
            
            for state in self.V.keys():
                action_values = []
                for action in range(self.env.action_space.n):
                    next_state, reward, done = self._get_transition_prob_reward(state, action)
                    
                    if done:
                        val = reward
                    else:
                        val = reward + self.gamma * self.V[next_state]
                    action_values.append(val)
                
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
        
        # *** تغییر مهم: برگرداندن تعداد مراحل برای فایل بنچمارک ***
        return iterations

    def _extract_policy(self):
        """استخراج سیاست بهینه (Policy) بر اساس مقادیر V همگرا شده"""
        for state in self.V.keys():
            action_values = []
            for action in range(self.env.action_space.n):
                next_state, reward, done = self._get_transition_prob_reward(state, action)
                if done:
                    val = reward
                else:
                    val = reward + self.gamma * self.V[next_state]
                action_values.append(val)
                
            best_action = np.argmax(action_values)
            self.policy[state] = best_action

    def get_action(self, state):
        """بازگرداندن اکشن بر اساس سیاست پیدا شده (برای محیط‌های تست و GUI)"""
        return self.policy.get(state, 0)