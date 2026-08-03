import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import glob

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent

os.makedirs('results/figures', exist_ok=True)

def plot_visitation_heatmap(env, agent, episodes=50, filename="results/figures/visitation_heatmap.png"):
    visits = np.zeros((env.grid_size, env.grid_size))
    for ep in range(episodes):
        state = env.reset()
        visits[state[0], state[1]] += 1
        for step in range(200):
            q_values = [agent.get_q(state, a) for a in range(4)]
            action = np.argmax(q_values) if any(q_values) else 0
            next_state, reward, done, _ = env.step(action)
            visits[next_state[0], next_state[1]] += 1
            state = next_state
            if done: break
                
    mask = np.zeros_like(visits, dtype=bool)
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL: mask[r, c] = True
                
    plt.figure(figsize=(8, 6))
    sns.heatmap(visits, mask=mask, cmap='YlOrRd', annot=False)
    plt.title("Q-Learning: State Visitation Heatmap")
    plt.savefig(filename)
    plt.close()

def plot_policy_difference(env, q_table1, q_table2, filename="results/figures/policy_difference.png"):
    diff_grid = np.zeros((env.grid_size, env.grid_size))
    mask = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL:
                mask[r, c] = True
                continue
            
            q1 = [q_table1.get(((r, c, 0, 0), a), 0.0) for a in range(4)]
            a1 = np.argmax(q1) if any(q1) else 0
            
            q2 = [q_table2.get(((r, c, 0, 0), a), 0.0) for a in range(4)]
            a2 = np.argmax(q2) if any(q2) else 0
            
            diff_grid[r, c] = 0 if a1 == a2 else 1
                
    plt.figure(figsize=(8, 6))
    cmap = ListedColormap(['#2ca02c', '#d62728'])
    sns.heatmap(diff_grid, mask=mask, cmap=cmap, cbar=False, linewidths=0.5, linecolor='black')
    
    import matplotlib.patches as mpatches
    same_patch = mpatches.Patch(color='#2ca02c', label='Same Policy')
    diff_patch = mpatches.Patch(color='#d62728', label='Different Policy')
    plt.legend(handles=[same_patch, diff_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title("Policy Difference: Q-Learning vs SARSA")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_value_heatmap(env, vi_agent, filename="results/figures/value_heatmap.png"):
    """رسم نقشه حرارتی ارزش‌های محاسبه شده توسط Value Iteration"""
    v_grid = np.zeros((env.grid_size, env.grid_size))
    mask = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL:
                mask[r, c] = True
            else:
                # گرفتن ارزش برای حالت بدون کلید و مانع پیش‌فرض
                v_grid[r, c] = vi_agent.V.get((r, c, 0, 0), 0.0)
                
    plt.figure(figsize=(8, 6))
    sns.heatmap(v_grid, mask=mask, cmap='viridis', annot=False)
    plt.title("Value Heatmap (Value Iteration)")
    plt.savefig(filename)
    plt.close()

def plot_transfer_learning_curves():
    """رسم نمودارهای مقایسه سناریوهای انتقال یادگیری از روی فایل‌های CSV"""
    for env_type in ['Similar', 'Different']:
        plt.figure(figsize=(10, 6))
        
        # پیدا کردن فایل‌های مربوط به این نوع محیط
        files = glob.glob(f"results/raw_data/transfer_{env_type}_*.csv")
        if not files:
            continue
            
        for file in files:
            # استخراج نام سناریو از اسم فایل
            scenario_name = os.path.basename(file).replace(f"transfer_{env_type}_", "").replace(".csv", "")
            
            df = pd.read_csv(file)
            # میانگین متحرک برای صاف کردن نمودار (Smoothing)
            smoothed_rewards = df['Reward'].rolling(window=15, min_periods=1).mean()
            plt.plot(df['Episode'], smoothed_rewards, label=scenario_name, alpha=0.8)
            
        plt.title(f"Transfer Learning Performance: {env_type} Environment")
        plt.xlabel("Episode")
        plt.ylabel("Moving Average of Reward")
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"results/figures/Transfer_Curves_{env_type}.png")
        plt.close()

def main():
    print("1. Generating algorithmic heatmaps...")
    env = DynamicMazeEnv()
    
    ql_agent = QLearningAgent(env, decay_type='exponential')
    ql_agent.train(episodes=400)
    plot_visitation_heatmap(env, ql_agent)
    
    sarsa_agent = SarsaLambdaAgent(env, lmbda=0.9, trace_type='replacing')
    sarsa_agent.train(episodes=400)
    plot_policy_difference(env, ql_agent.Q, sarsa_agent.Q)
    
    vi_agent = ValueIterationAgent(env)
    for method in ['train', 'solve', 'value_iteration', 'run', 'optimize']:
        if hasattr(vi_agent, method):
            getattr(vi_agent, method)()
            break
    plot_value_heatmap(env, vi_agent)
    
    print("2. Generating Transfer Learning curves from CSV logs...")
    plot_transfer_learning_curves()
    
    print("Analysis complete. ALL required figures are in 'results/figures/' folder.")

if __name__ == "__main__":
    main()