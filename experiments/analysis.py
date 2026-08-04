import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent

def load_settings():
    settings_path = os.path.join(os.path.dirname(__file__), 'configs', 'settings.json')
    with open(settings_path, 'r') as f:
        return json.load(f)

def plot_heatmap(data, title, filename, cmap='viridis'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, cmap=cmap, cbar=False, annot=False)
    plt.title(title)
    plt.savefig(f'results/figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def run_base_analysis():
    print("--- Loading settings.json ---")
    settings = load_settings()
    env = DynamicMazeEnv(use_reward_shaping=False)
    grid_size = env.grid_size

    # ==========================================
    # 1. اجرای Value Iteration و نقشه حرارتی ارزش
    # ==========================================
    print("1. Running Value Iteration and generating Value Heatmap...")
    vi = ValueIterationAgent(env, gamma=settings['gamma'])
    vi.run()

    v_grid = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            if env.grid[r, c] != env.WALL:
                v_grid[r, c] = vi.V.get((r, c, 0, 0), 0)
    # جایگزین کردن عکس یتیم قدیمی با دیتای واقعی پروژه
    plot_heatmap(v_grid, 'Value Iteration - V Table Heatmap', 'value_heatmap.png')

    # ==========================================
    # 2. اجرای Q-Learning و ردیابی نقشه بازدید
    # ==========================================
    print("2. Running Q-Learning and tracking state visitation...")
    visitation_counts = np.zeros((grid_size, grid_size))
    
    # استفاده از Wrapper برای ردیابی تعداد مراجعه بدون دستکاری کد محیط
    original_step = env.step
    def step_wrapper(action):
        r, c = env.agent_pos
        visitation_counts[r, c] += 1
        return original_step(action)
    env.step = step_wrapper

    ql = QLearningAgent(env, alpha=settings['alpha'], gamma=settings['gamma'],
                        epsilon_start=settings['epsilon_start'], epsilon_end=settings['epsilon_end'])
    ql_rewards, ql_steps, _, _ = ql.train(episodes=settings['episodes'], max_steps=settings['max_steps'])

    # جایگزین کردن عکس یتیم قدیمی با دیتای واقعی
    plot_heatmap(visitation_counts, 'State Visitation Heatmap (Q-Learning)', 'visitation_heatmap.png', cmap='hot')
    env.step = original_step # بازگرداندن محیط به حالت عادی

    # ==========================================
    # 3. اجرای SARSA(lambda)
    # ==========================================
    print("3. Running SARSA(lambda) evaluation...")
    sarsa = SarsaLambdaAgent(env, alpha=settings['alpha'], gamma=settings['gamma'],
                             lmbda=settings['sarsa_lambdas'][3], # Lambda = 0.9
                             epsilon_start=settings['epsilon_start'], epsilon_end=settings['epsilon_end'])
    sarsa_rewards, sarsa_steps, _, _ = sarsa.train(episodes=settings['episodes'], max_steps=settings['max_steps'])

    # ==========================================
    # 4. تولید منحنی‌های یادگیری گمشده (Base Learning Curves)
    # ==========================================
    print("4. Saving Base Learning Curves and Rewards Plot...")
    df = pd.DataFrame({
        'Episode': range(1, settings['episodes'] + 1),
        'QL_Rewards': ql_rewards,
        'SARSA_Rewards': sarsa_rewards,
        'QL_Steps': ql_steps,
        'SARSA_Steps': sarsa_steps
    })
    df.to_csv('results/raw_data/base_learning_curves.csv', index=False)

    window = 20
    plt.figure(figsize=(10, 6))
    plt.plot(pd.Series(ql_rewards).rolling(window).mean(), label='Q-Learning', color='blue')
    plt.plot(pd.Series(sarsa_rewards).rolling(window).mean(), label='SARSA (lambda=0.9)', color='red')
    plt.title(f'Base Algorithms Learning Curves (Moving Average {window})')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/base_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 5. مقایسه سیاست‌ها (Policy Match)
    # ==========================================
    print("5. Generating Policy Difference Heatmaps (VI vs QL / VI vs SARSA)...")
    ql_diff = np.zeros((grid_size, grid_size))
    sarsa_diff = np.zeros((grid_size, grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            if env.grid[r, c] == env.WALL:
                ql_diff[r, c] = -1
                sarsa_diff[r, c] = -1
                continue

            state = (r, c, 0, 0)
            vi_action = vi.get_action(state)

            # استخراج اکشن برتر Q-Learning
            ql_q_vals = [ql.get_q(state, a) for a in range(4)]
            ql_action = np.argmax(ql_q_vals) if any(q != 0 for q in ql_q_vals) else 0
            ql_diff[r, c] = 1 if vi_action == ql_action else 0

            # استخراج اکشن برتر SARSA
            sarsa_q_vals = [sarsa.get_q(state, a) for a in range(4)]
            sarsa_action = np.argmax(sarsa_q_vals) if any(q != 0 for q in sarsa_q_vals) else 0
            sarsa_diff[r, c] = 1 if vi_action == sarsa_action else 0

    cmap_diff = ListedColormap(['#303030', '#e74c3c', '#2ecc71']) # خاکستری (دیوار)، قرمز (تفاوت)، سبز (تطابق)

    plt.figure(figsize=(8, 6))
    sns.heatmap(ql_diff, cmap=cmap_diff, cbar=False, annot=False)
    plt.title('Policy Match: Q-Learning vs Value Iteration (Green=Match)')
    plt.savefig('results/figures/Policy_Diff_QL_vs_VI.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(sarsa_diff, cmap=cmap_diff, cbar=False, annot=False)
    plt.title('Policy Match: SARSA vs Value Iteration (Green=Match)')
    plt.savefig('results/figures/Policy_Diff_SARSA_vs_VI.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ Base analysis fully completed! All required data and figures generated based on settings.json.")

if __name__ == "__main__":
    run_base_analysis()