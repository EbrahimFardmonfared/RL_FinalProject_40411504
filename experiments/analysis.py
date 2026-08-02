import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent

# ساخت پوشه استاندارد برای ذخیره عکس نمودارها
os.makedirs('results/figures', exist_ok=True)

def plot_visitation_heatmap(env, agent, episodes=50, filename="results/figures/visitation_heatmap.png"):
    """رسم نقشه حرارتی از تعداد دفعات بازدید عامل از هر خانه و ذخیره آن"""
    visits = np.zeros((env.grid_size, env.grid_size))
    
    for ep in range(episodes):
        state = env.reset()
        visits[state[0], state[1]] += 1
        for step in range(200):
            q_values = [agent.get_q(state, a) for a in range(4)]
            max_q = max(q_values)
            best_actions = [a for a in range(4) if q_values[a] == max_q]
            action = best_actions[0]
            
            next_state, reward, done, _ = env.step(action)
            visits[next_state[0], next_state[1]] += 1
            state = next_state
            if done:
                break
                
    mask = np.zeros_like(visits, dtype=bool)
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL:
                mask[r, c] = True
                
    plt.figure(figsize=(8, 6))
    sns.heatmap(visits, mask=mask, cmap='YlOrRd', annot=False)
    plt.title("Q-Learning: State Visitation Heatmap")
    plt.savefig(filename)
    print(f"Heatmap saved to {filename}")
    plt.close()

def plot_policy_difference(env, q_table1, q_table2, filename="results/figures/policy_difference.png"):
    """مقایسه سیاست دو عامل مختلف و ذخیره تصویر آن"""
    diff_grid = np.zeros((env.grid_size, env.grid_size))
    mask = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL:
                mask[r, c] = True
                continue
                
            q1 = [q_table1.get(((r, c), a), 0.0) for a in range(4)]
            a1 = np.argmax(q1) if any(q1) else 0
            
            q2 = [q_table2.get(((r, c), a), 0.0) for a in range(4)]
            a2 = np.argmax(q2) if any(q2) else 0
            
            if a1 != a2:
                diff_grid[r, c] = 1
            else:
                diff_grid[r, c] = 0
                
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
    print(f"Policy Difference map saved to {filename}")
    plt.close()

def main():
    print("Training agents for visual analysis...")
    env = DynamicMazeEnv()
    
    ql_agent = QLearningAgent(env, decay_type='exponential')
    ql_agent.train(episodes=500)
    
    plot_visitation_heatmap(env, ql_agent)
    
    sarsa_agent = SarsaLambdaAgent(env, lmbda=0.9, trace_type='replacing')
    sarsa_agent.train(episodes=500)
    
    plot_policy_difference(env, ql_agent.Q, sarsa_agent.Q)
    print("Analysis complete. Check the 'results/figures' folder for images.")

if __name__ == "__main__":
    main()