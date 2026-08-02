import time

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-4):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {}
        self.policy = {}
        self.states = self.env.get_all_states()
        
        # مقداردهی اولیه تابع ارزش با صفر برای تمام حالت‌ها
        for s in self.states:
            self.V[s] = 0.0
            
    def run(self):
        """اجرای حلقه اصلی الگوریتم ارزیابی ارزش (Value Iteration)"""
        start_time = time.time()
        iterations = 0
        
        while True:
            delta = 0
            new_V = self.V.copy()
            
            for s in self.states:
                # اگر در حالت پایانی (هدف) هستیم، ارزش صفر می‌ماند
                if (s[0], s[1]) == self.env.goal_pos:
                    new_V[s] = 0.0
                    continue
                    
                action_values = []
                for a in range(4): # 0: بالا، 1: راست، 2: پایین، 3: چپ
                    val = 0
                    # P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
                    for prob, next_s, reward, done in self.env.get_transitions(s, a):
                        if done:
                            val += prob * reward
                        else:
                            val += prob * (reward + self.gamma * self.V[next_s])
                    action_values.append(val)
                
                # بروزرسانی بلمن
                best_action_value = max(action_values)
                delta = max(delta, abs(self.V[s] - best_action_value))
                new_V[s] = best_action_value
                
            self.V = new_V
            iterations += 1
            
            # شرط همگرایی
            if delta < self.theta:
                break
                
        execution_time = time.time() - start_time
        self._extract_policy()
        
        print(f"Value Iteration converged in {iterations} iterations.")
        print(f"Execution time: {execution_time:.4f} seconds for Gamma = {self.gamma}")
        
        return iterations, execution_time

    def _extract_policy(self):
        """استخراج سیاست حریصانه (Greedy Policy) از روی تابع ارزش بهینه"""
        for s in self.states:
            if (s[0], s[1]) == self.env.goal_pos:
                self.policy[s] = None
                continue
                
            action_values = []
            for a in range(4):
                val = 0
                for prob, next_s, reward, done in self.env.get_transitions(s, a):
                    if done:
                        val += prob * reward
                    else:
                        val += prob * (reward + self.gamma * self.V[next_s])
                action_values.append(val)
                
            # انتخاب بهترین عمل
            best_action = action_values.index(max(action_values))
            self.policy[s] = best_action
            
    def get_action(self, state):
        return self.policy.get(state, random.choice([0,1,2,3]))