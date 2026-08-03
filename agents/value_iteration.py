import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # V-table: فضای حالت ۴ بعدی
        self.V = {}
        self.policy = {}
        
        # 1. مقداردهی اولیه به حالت‌ها
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if env.grid[r, c] == env.WALL:
                    continue
                for has_key in [0, 1]:
                    for patrol_idx in range(len(env.patrol_route)):
                        self.V[(r, c, has_key, patrol_idx)] = 0.0

        # 2. پیش‌محاسبه ماتریس انتقال (Caching) برای افزایش 1000 برابری سرعت
        self.P = {}
        self._build_transition_model()

    def _build_transition_model(self):
        """ساخت ماتریس انتقال: تمام احتمالات محیط فقط یک بار محاسبه و در حافظه ذخیره می‌شوند"""
        # رفع ارور tuple: نیازی به copy() نیست چون شیء مستقیماً جایگزین می‌شود
        original_pos = self.env.agent_pos if hasattr(self.env, 'agent_pos') else None
        original_key = getattr(self.env, 'has_key', False)
        original_patrol = getattr(self.env, 'patrol_idx', 0)
        original_steps = getattr(self.env, 'steps', 0)

        for state in self.V.keys():
            self.P[state] = {}
            for action in range(4):
                # قراردادن محیط در حالت مشخص
                self.env.agent_pos = [state[0], state[1]]
                self.env.has_key = bool(state[2])
                self.env.patrol_idx = state[3]
                
                # صفر کردن قدم‌شمار برای جلوگیری از TimeOut شدن ناخواسته
                if hasattr(self.env, 'steps'):
                    self.env.steps = 0
                    
                next_state, reward, done, _ = self.env.step(action)
                self.P[state][action] = (next_state, reward, done)

        # بازگرداندن محیط به وضعیت اورجینال
        if original_pos is not None:
            self.env.agent_pos = original_pos
        self.env.has_key = original_key
        self.env.patrol_idx = original_patrol
        if hasattr(self.env, 'steps'):
            self.env.steps = original_steps

    def run(self):
        """اجرای فوق‌سریع الگوریتم Value Iteration با استفاده از مقادیر Cache شده"""
        iterations = 0
        while True:
            delta = 0
            new_V = self.V.copy()
            
            for state in self.V.keys():
                action_values = []
                for action in range(4): 
                    # فراخوانی سریع از مموری به جای شبیه‌سازی سنگین
                    next_state, reward, done = self.P[state][action]
                    
                    if done:
                        val = reward
                    else:
                        val = reward + self.gamma * self.V[next_state]
                    action_values.append(val)
                
                best_value = max(action_values)
                new_V[state] = best_value
                delta = max(delta, abs(best_value - self.V[state]))
            
            self.V = new_V
            iterations += 1
            
            if delta < self.theta:
                break
                
        self._extract_policy()
        return iterations

    def _extract_policy(self):
        """استخراج سیاست بهینه بر اساس مقادیر V همگرا شده"""
        for state in self.V.keys():
            action_values = []
            for action in range(4):
                next_state, reward, done = self.P[state][action]
                if done:
                    val = reward
                else:
                    val = reward + self.gamma * self.V[next_state]
                action_values.append(val)
                
            self.policy[state] = int(np.argmax(action_values))

    def get_action(self, state):
        """بازگرداندن اکشن بر اساس سیاست پیدا شده"""
        return self.policy.get(state, 0)