import sys
import os
import csv
import pickle

# اضافه کردن مسیر پوشه اصلی به سیستم برای شناسایی ماژول‌ها
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from transfer.transfer_learning import TransferExperiment

def save_logs_to_csv(scenario_name, env_type, logs):
    """تابع کمکی برای ذخیره داده‌های خام هر سناریو در پوشه results/raw_data"""
    clean_name = scenario_name.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').strip()
    file_path = f"results/raw_data/transfer_{env_type}_{clean_name}.csv"
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # اضافه کردن ستون‌های گام‌ها، برخورد با دیوار و جریمه‌ها طبق خواسته داکیومنت
        writer.writerow(['Episode', 'Reward', 'Success', 'Steps', 'Wall_Hits', 'Penalty_Hits'])
        for ep in range(len(logs['rewards'])):
            writer.writerow([
                ep + 1, 
                logs['rewards'][ep], 
                logs['success'][ep],
                logs['steps'][ep],
                logs['wall_hits'][ep],
                logs['penalty_hits'][ep]
            ])

def evaluate_logs(scenario_name, env_type, logs):
    """Helper function to print performance summary and save to CSV"""
    init_perf = sum(logs['rewards'][:10]) / 10
    final_perf = sum(logs['rewards'][-10:]) / 10
    total_success = sum(logs['success'])
    print(f"    -> {scenario_name:<20} | Initial Perf: {init_perf:>7.1f} | Final Perf: {final_perf:>7.1f} | Total Success: {total_success:>4}")
    
    save_logs_to_csv(scenario_name, env_type, logs)

def main():
    os.makedirs("results/raw_data", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    print("1. Creating Source Environment and Training Base Agent...")
    base_env = DynamicMazeEnv()
    source_agent = QLearningAgent(base_env, decay_type='exponential')
    
    source_agent.train(episodes=500)
    source_q_table = source_agent.Q
    
    model_path = "results/models/base_q_learning_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(source_q_table, f)
    print(f"  [Base Agent Trained and Model Saved to {model_path}]\n")

    print("2. Generating Target Environments (Similar & Different)...")
    similar_env = base_env.generate_similar_map()
    different_env = base_env.generate_different_map()
    print("  [Target Environments Ready]\n")

    episodes = 300 
    betas = [0.25, 0.50, 0.75] # تست تمام مقادیر بتای خواسته شده

    # ==========================================
    # آزمایش اول: محیط مشابه
    # ==========================================
    print("="*60)
    print("--- TRANSFER TO SIMILAR ENVIRONMENT ---")
    exp_sim = TransferExperiment(base_env, similar_env, source_q_table)
    
    logs_sim_1 = exp_sim.run_scenario_1_scratch(episodes)
    evaluate_logs("Scenario_1_Scratch", "Similar", logs_sim_1)
    
    logs_sim_2 = exp_sim.run_scenario_2_full_transfer(episodes)
    evaluate_logs("Scenario_2_Full", "Similar", logs_sim_2)
    
    for b in betas:
        logs_sim_3 = exp_sim.run_scenario_3_beta_transfer(b, episodes)
        evaluate_logs(f"Scenario_3_B_{b}", "Similar", logs_sim_3)
    
    logs_sim_4 = exp_sim.run_scenario_4_selective_transfer(episodes)
    evaluate_logs("Scenario_4_Select", "Similar", logs_sim_4)

    # ==========================================
    # آزمایش دوم: محیط متفاوت
    # ==========================================
    print("\n" + "="*60)
    print("--- TRANSFER TO DIFFERENT ENVIRONMENT ---")
    exp_diff = TransferExperiment(base_env, different_env, source_q_table)
    
    logs_diff_1 = exp_diff.run_scenario_1_scratch(episodes)
    evaluate_logs("Scenario_1_Scratch", "Different", logs_diff_1)
    
    logs_diff_2 = exp_diff.run_scenario_2_full_transfer(episodes)
    evaluate_logs("Scenario_2_Full", "Different", logs_diff_2)
    
    for b in betas:
        logs_diff_3 = exp_diff.run_scenario_3_beta_transfer(b, episodes)
        evaluate_logs(f"Scenario_3_B_{b}", "Different", logs_diff_3)
    
    logs_diff_4 = exp_diff.run_scenario_4_selective_transfer(episodes)
    evaluate_logs("Scenario_4_Select", "Different", logs_diff_4)

    print("\nTransfer Learning Experiments Completed Successfully!")
    print("All raw logs are saved in 'results/raw_data/' in CSV format.")

if __name__ == "__main__":
    main()