import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# افزودن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent

def get_full_policy(agent, env, is_vi=False):
    """استخراج سیاست کامل برای مقایسه"""
    policy = {}
    default_p = 0
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL:
                continue
            for has_key in [0, 1]:
                state = (r, c, has_key, default_p)
                if is_vi:
                    if hasattr(agent, 'get_action'):
                        policy[state] = agent.get_action(state)
                    elif hasattr(agent, 'policy') and isinstance(agent.policy, dict):
                        policy[state] = agent.policy.get(state, 0)
                else:
                    q_values = [agent.get_q(state, a) for a in range(4)]
                    if any(q != 0 for q in q_values):
                        policy[state] = np.argmax(q_values)
                    else:
                        policy[state] = 0
    return policy

def run_base_experiments():
    # ایجاد پوشه‌های نتایج در صورت عدم وجود
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/raw_data', exist_ok=True)
    
    episodes = 500
    env = DynamicMazeEnv(use_reward_shaping=False)
    
    # 1. اجرای Value Iteration به عنوان مرجع (Reference)
    print("\n--- Training Value Iteration (Reference) ---")
    vi = ValueIterationAgent(env, gamma=0.9)
    vi_executed = False
    
    # *** اصلاح بحرانی: اضافه شدن 'run' به لیست متدها ***
    for method in ['run', 'solve', 'train', 'value_iteration']:
        if hasattr(vi, method):
            getattr(vi, method)()
            print(f"Value Iteration successfully executed using method: {method}")
            vi_executed = True
            break
            
    if not vi_executed:
        print("Warning: Value Iteration could not be run. Check method names.")
        
    vi_policy = get_full_policy(vi, env, is_vi=True)
    
    # 2. آموزش Q-Learning
    print("\n--- Training Q-Learning ---")
    ql = QLearningAgent(env)
    ql_rewards, ql_steps, _, _ = ql.train(episodes)
    ql_policy = get_full_policy(ql, env)
    
    # مقایسه سیاست QL با VI
    diff_count = sum(1 for s in vi_policy if vi_policy[s] != ql_policy.get(s, 0))
    total_states = len(vi_policy)
    match_percentage = ((total_states - diff_count) / total_states) * 100 if total_states > 0 else 0
    print(f"Q-Learning Policy Match with VI: {match_percentage:.2f}%")

    # 3. آموزش SARSA(lambda) با مقادیر مختلف λ (اصلاح شده)
    lambdas = [0.0, 0.3, 0.7, 0.9]
    sarsa_results = {}
    
    for l in lambdas:
        print(f"\n--- Training SARSA(lambda={l}) ---")
        sarsa = SarsaLambdaAgent(env, lmbda=l)
        r, s, _, _ = sarsa.train(episodes)
        sarsa_results[l] = {'rewards': r, 'steps': s}

    # 4. ذخیره داده‌های خام (CSV)
    data = {
        'Episode': range(1, episodes + 1),
        'QL_Rewards': ql_rewards,
        'QL_Steps': ql_steps
    }
    for l in lambdas:
        data[f'SARSA_L{l}_Rewards'] = sarsa_results[l]['rewards']
        data[f'SARSA_L{l}_Steps'] = sarsa_results[l]['steps']
        
    df = pd.DataFrame(data)
    df.to_csv('results/raw_data/base_learning_curves.csv', index=False)
    print("\nData saved to 'results/raw_data/base_learning_curves.csv'")

    # 5. رسم نمودار پاداش‌ها (Moving Average)
    window = 20
    plt.figure(figsize=(10, 6))
    
    plt.plot(pd.Series(ql_rewards).rolling(window).mean(), label='Q-Learning', linewidth=2)
    for l in lambdas:
        plt.plot(pd.Series(sarsa_results[l]['rewards']).rolling(window).mean(), label=f'SARSA (λ={l})', alpha=0.8)
        
    plt.title(f'Learning Curves (Moving Average {window})')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/figures/base_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved to 'results/figures/base_rewards.png'")

if __name__ == "__main__":
    run_base_experiments()