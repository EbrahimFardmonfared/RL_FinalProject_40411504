from environments.maze import DynamicMazeEnv
from agents.value_iteration import ValueIterationAgent

def main():
    print("Initializing Dynamic Maze Environment...")
    env = DynamicMazeEnv()
    print(f"Maze created successfully with Seed = {env.seed}")
    
    # تست 3 ضریب تنزیل مختلف طبق خواسته داکیومنت
    gamma_values = [0.7, 0.9, 0.99]
    
    for gamma in gamma_values:
        print(f"\n{'-'*40}")
        print(f"Running Value Iteration with Gamma = {gamma} ...")
        
        agent = ValueIterationAgent(env, gamma=gamma)
        iterations, exec_time = agent.run()
        
        # این اطلاعات برای نوشتن گزارش تحلیلی نهایی بسیار مهم هستند
        print(f"Finished in {iterations} iterations.")
        print(f"Time taken: {exec_time:.4f} seconds.")

if __name__ == "__main__":
    main()