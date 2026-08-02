from environments.maze import DynamicMazeEnv
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent

def main():
    print("Initializing Dynamic Maze Environment...")
    env = DynamicMazeEnv()
    print(f"Maze created successfully with Seed = {env.seed}")
    
    # ==========================================
    # بخش اول: اجرای Value Iteration
    # ==========================================
    print("\n" + "="*50)
    print("--- Running Value Iteration ---")
    gamma_values = [0.7, 0.9, 0.99]
    for gamma in gamma_values:
        print(f"\nGamma = {gamma}:")
        vi_agent = ValueIterationAgent(env, gamma=gamma)
        iterations, exec_time = vi_agent.run()
        
    # ==========================================
    # بخش دوم: اجرای Q-Learning
    # ==========================================
    print("\n" + "="*50)
    print("--- Running Q-Learning ---")
    
    episodes = 500 # تعداد اپیزودها برای آموزش
    
    # تست 1: کاهش نمایی (Exponential Decay)
    print(f"\nTraining Q-Learning with Exponential Decay ({episodes} episodes)...")
    ql_exp = QLearningAgent(env, decay_type='exponential')
    logs_exp = ql_exp.train(episodes=episodes)
    # محاسبه میانگین پاداش در 10 اپیزود آخر برای بررسی میزان یادگیری
    avg_reward_exp = sum(logs_exp['rewards'][-10:]) / 10
    total_success_exp = sum(logs_exp['success'])
    print(f"Final 10 episodes average reward: {avg_reward_exp:.2f}")
    print(f"Total successful episodes (reached goal): {total_success_exp}")
    
    # تست 2: کاهش خطی (Linear Decay)
    print(f"\nTraining Q-Learning with Linear Decay ({episodes} episodes)...")
    ql_linear = QLearningAgent(env, decay_type='linear')
    logs_linear = ql_linear.train(episodes=episodes)
    avg_reward_linear = sum(logs_linear['rewards'][-10:]) / 10
    total_success_linear = sum(logs_linear['success'])
    print(f"Final 10 episodes average reward: {avg_reward_linear:.2f}")
    print(f"Total successful episodes (reached goal): {total_success_linear}")

if __name__ == "__main__":
    main()