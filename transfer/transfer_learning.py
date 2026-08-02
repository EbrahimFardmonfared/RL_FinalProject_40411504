import copy
from agents.q_learning import QLearningAgent

def get_local_neighborhood(env, r, c):
    """
    استخراج ساختار محلی (یک شبکه 3x3) اطراف یک خانه مشخص.
    این تابع برای تشخیص تغییرات محلی در سناریوی انتقال انتخابی استفاده می‌شود.
    """
    neighborhood = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.grid_size and 0 <= nc < env.grid_size:
                neighborhood.append(env.grid[nr, nc])
            else:
                # اگر خارج از مرز باشد، آن را معادل دیوار فرض می‌کنیم
                neighborhood.append(env.WALL)
    return neighborhood

def is_neighborhood_changed(env_source, env_target, r, c):
    """بررسی اینکه آیا همسایگی محلی یک حالت بین دو محیط تغییر کرده است یا خیر"""
    source_nb = get_local_neighborhood(env_source, r, c)
    target_nb = get_local_neighborhood(env_target, r, c)
    return source_nb != target_nb

class TransferExperiment:
    def __init__(self, source_env, target_env, source_q_table):
        self.source_env = source_env
        self.target_env = target_env
        self.source_q_table = source_q_table

    def run_scenario_1_scratch(self, episodes=300):
        """سناریو 1: آموزش از صفر (خط مبنا)"""
        print("  -> Running Scenario 1: Learning from scratch...")
        agent = QLearningAgent(self.target_env, decay_type='exponential')
        return agent.train(episodes=episodes)

    def run_scenario_2_full_transfer(self, episodes=300):
        """سناریو 2: انتقال کامل تمام جدول Q محیط مبدأ"""
        print("  -> Running Scenario 2: Full Transfer...")
        agent = QLearningAgent(self.target_env, decay_type='exponential')
        agent.Q = copy.deepcopy(self.source_q_table)
        return agent.train(episodes=episodes)

    def run_scenario_3_beta_transfer(self, beta, episodes=300):
        """سناریو 3: انتقال تعدیل‌شده با ضریب بتا"""
        print(f"  -> Running Scenario 3: Beta Transfer (beta={beta})...")
        agent = QLearningAgent(self.target_env, decay_type='exponential')
        agent.Q = {state_action: q_val * beta for state_action, q_val in self.source_q_table.items()}
        return agent.train(episodes=episodes)

    def run_scenario_4_selective_transfer(self, episodes=300):
        """سناریو 4: انتقال انتخابی (فقط حالت‌هایی که همسایگی آنها تغییر نکرده است)"""
        print("  -> Running Scenario 4: Selective Transfer (Local Check)...")
        agent = QLearningAgent(self.target_env, decay_type='exponential')
        transferred_states = 0
        
        for (state, action), q_val in self.source_q_table.items():
            r, c = state[0], state[1]
            if not is_neighborhood_changed(self.source_env, self.target_env, r, c):
                agent.Q[(state, action)] = q_val
                transferred_states += 1
                
        # یک پرینت کوچک برای تحلیل در گزارش
        print(f"     [Info] Transferred {transferred_states} state-action pairs selectively.")
        return agent.train(episodes=episodes)