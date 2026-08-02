from environments.maze import DynamicMazeEnv
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent

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
    episodes = 500
    
    print(f"\nTraining Q-Learning with Exponential Decay ({episodes} episodes)...")
    ql_exp = QLearningAgent(env, decay_type='exponential')
    logs_exp = ql_exp.train(episodes=episodes)
    print(f"Final 10 episodes avg reward: {sum(logs_exp['rewards'][-10:])/10:.2f} | Total Successes: {sum(logs_exp['success'])}")
    
    print(f"\nTraining Q-Learning with Linear Decay ({episodes} episodes)...")
    ql_linear = QLearningAgent(env, decay_type='linear')
    logs_linear = ql_linear.train(episodes=episodes)
    print(f"Final 10 episodes avg reward: {sum(logs_linear['rewards'][-10:])/10:.2f} | Total Successes: {sum(logs_linear['success'])}")

    # ==========================================
    # بخش سوم: اجرای SARSA(lambda)
    # ==========================================
    print("\n" + "="*50)
    print("--- Running SARSA(lambda) ---")
    
    # تست مقادیر مختلف لامبدا طبق خواسته داکیومنت
    lambda_values = [0.7, 0.9, 0.99]
    
    for lmbda in lambda_values:
        print(f"\nTraining SARSA(lambda={lmbda}) with Replacing Trace ({episodes} episodes)...")
        sarsa_agent = SarsaLambdaAgent(env, lmbda=lmbda, trace_type='replacing')
        logs_sarsa = sarsa_agent.train(episodes=episodes)
        print(f"Final 10 episodes avg reward: {sum(logs_sarsa['rewards'][-10:])/10:.2f} | Total Successes: {sum(logs_sarsa['success'])}")

if __name__ == "__main__":
    main()