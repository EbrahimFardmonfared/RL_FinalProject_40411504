import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.value_iteration import ValueIterationAgent

def run_benchmarks():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/raw_data', exist_ok=True)
    
    episodes = 500
    env = DynamicMazeEnv(use_reward_shaping=False)
    
    # 1. VI Convergence
    print("\n--- Running Value Iteration Convergence Benchmark ---")
    gammas = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    vi_results = []
    for g in gammas:
        print(f"Evaluating VI with Gamma = {g}...")
        vi = ValueIterationAgent(env, gamma=g)
        start_time = time.time()
        iterations = vi.run()
        elapsed_time = time.time() - start_time
        vi_results.append({'Gamma': g, 'Iterations': iterations, 'Time_Seconds': round(elapsed_time, 4)})
        
    df_vi = pd.DataFrame(vi_results)
    df_vi.to_csv('results/raw_data/vi_convergence_vs_gamma.csv', index=False)
    
    plt.figure(figsize=(8, 5))
    plt.plot(df_vi['Gamma'], df_vi['Time_Seconds'], marker='o', color='purple', linewidth=2)
    plt.title('Value Iteration Convergence Time vs Gamma')
    plt.xlabel('Gamma (Discount Factor)')
    plt.ylabel('Convergence Time (Seconds)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('results/figures/vi_convergence_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Q-Learning Epsilon Decay Comparison
    print("\n--- Running Q-Learning Epsilon Decay Comparison ---")
    ql_linear = QLearningAgent(env, decay_type='linear', decay_rate=0.002) 
    r_lin, _, _, _ = ql_linear.train(episodes, max_steps=500)
    
    ql_exp = QLearningAgent(env, decay_type='exponential', decay_rate=0.99) 
    r_exp, _, _, _ = ql_exp.train(episodes, max_steps=500)
    
    df_decay = pd.DataFrame({'Episode': range(1, episodes + 1), 'Linear_Reward': r_lin, 'Exponential_Reward': r_exp})
    df_decay.to_csv('results/raw_data/ql_epsilon_decay_comparison.csv', index=False)
    
    window = 20
    plt.figure(figsize=(10, 6))
    plt.plot(pd.Series(r_lin).rolling(window).mean(), label='Linear Decay', color='blue')
    plt.plot(pd.Series(r_exp).rolling(window).mean(), label='Exponential Decay', color='orange')
    plt.title(f'Q-Learning Epsilon Decay Comparison (Moving Average {window})')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/ql_epsilon_decay.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ Benchmarks completed! (SARSA evaluation handled dynamically in analysis.py)")

if __name__ == "__main__":
    run_benchmarks()