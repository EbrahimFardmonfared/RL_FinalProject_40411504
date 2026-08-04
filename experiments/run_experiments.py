import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from transfer.transfer_learning import TransferLearningExperiment
from agents.value_iteration import ValueIterationAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.q_learning import QLearningAgent

def save_model(model_data, filename):
    """ذخیره مدل‌ها در پوشه results/models"""
    os.makedirs('results/models', exist_ok=True)
    with open(f'results/models/{filename}', 'wb') as f:
        pickle.dump(model_data, f)

def plot_q_difference(source_q, target_q, target_name, grid_size):
    """رسم نقشه حرارتی تفاوت مقادیر Q قبل و بعد از انتقال"""
    diff_grid = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            state = (r, c, 0, 0)
            diffs = []
            for a in range(4):
                sq = source_q.get((state, a), 0.0)
                tq = target_q.get((state, a), 0.0)
                diffs.append(abs(tq - sq))
            diff_grid[r, c] = np.max(diffs) if diffs else 0.0

    plt.figure(figsize=(8, 6))
    sns.heatmap(diff_grid, cmap='viridis', annot=False)
    plt.title(f'Max Q-Value Difference (Source vs Target) - {target_name}')
    plt.savefig(f'results/figures/Q_Diff_Transfer_{target_name}.png', dpi=300)
    plt.close()

def run_all():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/raw_data', exist_ok=True)

    # 🔴 محیط به صورت نیتیو از شماره دانشجویی (seed=0, grid=15) استفاده می‌کند
    print("--- 1. Generating & Saving Baseline Models (VI & SARSA) ---")
    source_env = DynamicMazeEnv(use_reward_shaping=False)
    grid_size = source_env.grid_size
    
    vi = ValueIterationAgent(source_env)
    vi.run()
    save_model(vi.V, 'vi_vtable.pkl')
    
    sarsa = SarsaLambdaAgent(source_env, lmbda=0.9)
    sarsa.train(episodes=500, max_steps=500)
    save_model(sarsa.Q, 'sarsa_lambda_0.9_qtable.pkl')
    print("-> Reference Models Saved.")

    print("\n--- 2. Full Transfer Learning Experiments ---")
    
    # 🔴 رفع باگ: ذخیره خروجی توابع در متغیرهای جدید
    similar_env = source_env.generate_similar_map() if hasattr(source_env, 'generate_similar_map') else DynamicMazeEnv(use_reward_shaping=False)
    different_env = source_env.generate_different_map() if hasattr(source_env, 'generate_different_map') else DynamicMazeEnv(use_reward_shaping=False)

    episodes = 500
    for target_name, target_env in [("Similar", similar_env), ("Different", different_env)]:
        print(f"\nEvaluating {target_name} Environment...")
        exp = TransferLearningExperiment(source_env, target_env)
        
        # آموزش مبدأ و ذخیره دانش پایه
        exp.train_source(episodes, max_steps=500)
        save_model(exp.source_q_table, f'source_q_table_{target_name}.pkl')

        results = {}
        results['Scratch'] = exp.train_target_from_scratch(episodes, max_steps=500)
        results['Full'] = exp.train_target_full_transfer(episodes, max_steps=500)
        
        for beta in [0.25, 0.5, 0.75]:
            results[f'Beta_{beta}'] = exp.train_target_beta_transfer(episodes, max_steps=500, beta=beta)
        results['Selective'] = exp.train_target_selective_transfer(episodes, max_steps=500)

        # محاسبه هدف کامل برای رسم تفاوت Q-table
        print(f"Generating Q-Diff Heatmap for {target_name}...")
        agent_target = QLearningAgent(target_env)
        agent_target.Q = copy.deepcopy(exp.source_q_table)
        agent_target.epsilon = 0.5
        agent_target.train(episodes, max_steps=500)
        plot_q_difference(exp.source_q_table, agent_target.Q, target_name, grid_size)
        save_model(agent_target.Q, f'target_full_q_table_{target_name}.pkl')

        # ذخیره داده‌های CSV و رسم نمودار انتقال
        window = 20
        plt.figure(figsize=(10, 6))
        for method_name, data in results.items():
            rewards, steps, wall_hits, penalty_hits = data
            success = [1 if s < 500 else 0 for s in steps]
            
            df = pd.DataFrame({'Episode': range(1, len(rewards)+1), 'Rewards': rewards, 'Steps': steps, 'Success': success})
            clean_name = method_name.replace('.', '_')
            df.to_csv(f'results/raw_data/transfer_{target_name}_{clean_name}.csv', index=False)
            
            plt.plot(pd.Series(rewards).rolling(window).mean(), label=method_name)
        
        plt.title(f'Transfer Learning Performance - {target_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'results/figures/Transfer_Curves_{target_name}.png')
        plt.close()
        
    print("\n✅ Run Experiments Completed. CSVs, Figures, and Models Generated.")

if __name__ == "__main__":
    run_all()