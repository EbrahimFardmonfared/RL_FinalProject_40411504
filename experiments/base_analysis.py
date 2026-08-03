import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent

os.makedirs('results/raw_data', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

def run_base_experiments():
    env = DynamicMazeEnv()
    
    # === 1. تست Value Iteration و لاگ همگرایی ===
    print("Running Value Iteration...")
    vi = ValueIterationAgent(env)
    
    # حل محیط و محاسبه ارزش‌ها
    if hasattr(vi, 'solve'):
        vi.solve()
    elif hasattr(vi, 'train'):
        vi.train()
    elif hasattr(vi, 'value_iteration'):
        vi.value_iteration()
        
    print("Value Iteration complete. State values calculated.")
    
    # === 2. تست SARSA با لامبداهای خواسته شده در داکیومنت ===
    lambdas = [0.0, 0.3, 0.7, 0.9]
    sarsa_results = {}
    
    print("\nTesting SARSA(lambda) with various lambda values...")
    # ذخیره لاگ SARSA در CSV
    sarsa_log_file = "results/raw_data/sarsa_lambda_experiments.csv"
    with open(sarsa_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Lambda', 'Episode', 'Reward'])
        
        for lmb in lambdas:
            print(f"  Training SARSA(lambda={lmb})...")
            agent = SarsaLambdaAgent(env, lmbda=lmb, trace_type='replacing')
            rewards = agent.train(episodes=300)
            sarsa_results[lmb] = agent.Q
            
            # ذخیره رکوردهای این لامبدا
            for ep, r in enumerate(rewards):
                writer.writerow([lmb, ep + 1, r])

    print(f"SARSA experiments logged to: {sarsa_log_file}")

    # === 3. مقایسه سیاست Q-Learning با سیاست مرجع (VI) ===
    print("\nRunning Q-Learning for Policy Comparison...")
    ql = QLearningAgent(env, decay_type='exponential')
    ql.train(episodes=500)
    
    # تولید آرایه تفاوت سیاست‌ها
    diff_grid = np.zeros((env.grid_size, env.grid_size))
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL:
                continue
            
            # استخراج سیاست VI (مرجع) - گرفتن بهترین عمل بر اساس ارزش همسایه‌ها
            vi_a = 0
            if hasattr(vi, 'policy'):
                vi_a = vi.policy.get((r, c, 0, 0), 0)
            elif hasattr(vi, 'get_best_action'):
                vi_a = vi.get_best_action((r, c, 0, 0))
            
            # استخراج سیاست Q-Learning
            q_vals = [ql.get_q((r, c, 0, 0), a) for a in range(4)]
            ql_a = np.argmax(q_vals) if any(q_vals) else 0
            
            # اگر متفاوت بود 1، اگر مشابه بود 0
            diff_grid[r, c] = 0 if vi_a == ql_a else 1

    # رسم نمودار صحیح مقایسه سیاست با مرجع Value Iteration
    print("Generating Policy Difference Plot...")
    plt.figure(figsize=(8,6))
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#2ca02c', '#d62728'])
    sns.heatmap(diff_grid, cmap=cmap, cbar=False, linewidths=0.5, linecolor='black')
    
    import matplotlib.patches as mpatches
    same_patch = mpatches.Patch(color='#2ca02c', label='Same Policy (QL == VI)')
    diff_patch = mpatches.Patch(color='#d62728', label='Different Policy (QL != VI)')
    plt.legend(handles=[same_patch, diff_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Policy Difference: Q-Learning vs Value Iteration (Reference)")
    plt.tight_layout()
    plt.savefig("results/figures/Policy_Diff_QL_vs_VI.png")
    plt.close()
    
    print("\nAnalytics generated successfully! Check results/figures and results/raw_data.")

if __name__ == "__main__":
    run_base_experiments()