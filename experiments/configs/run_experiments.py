import sys
import os

# اضافه کردن مسیر پوشه اصلی به سیستم برای شناسایی ماژول‌ها
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from transfer.transfer_learning import TransferExperiment

def evaluate_logs(scenario_name, logs):
    """تابع کمکی برای چاپ خلاصه عملکرد جهت تحلیل در گزارش"""
    init_perf = sum(logs['rewards'][:10]) / 10
    final_perf = sum(logs['rewards'][-10:]) / 10
    total_success = sum(logs['success'])
    print(f"     -> {scenario_name} | عملكرد اوليه: {init_perf:.1f} | عملكرد نهايي: {final_perf:.1f} | مجموع موفقيت: {total_success}")

def main():
    print("1. Creating Source Environment and Training Base Agent...")
    base_env = DynamicMazeEnv()
    source_agent = QLearningAgent(base_env, decay_type='exponential')
    
    # آموزش عامل در محیط مبدأ برای 500 اپیزود تا به پایداری برسد
    source_agent.train(episodes=500)
    source_q_table = source_agent.Q
    print("   [Base Agent Trained Successfully]\n")

    print("2. Generating Target Environments (Similar & Different)...")
    similar_env = base_env.generate_similar_map()
    different_env = base_env.generate_different_map()
    print("   [Target Environments Ready]\n")

    episodes = 300 # تعداد اپیزود برای تست محیط‌های مقصد
    beta_value = 0.5 # تست ضریب بتا (مطابق درخواست داکیومنت)

    # ==========================================
    # آزمایش اول: محیط مشابه
    # ==========================================
    print("="*60)
    print("--- TRANSFER TO SIMILAR ENVIRONMENT ---")
    exp_sim = TransferExperiment(base_env, similar_env, source_q_table)
    
    logs_sim_1 = exp_sim.run_scenario_1_scratch(episodes)
    evaluate_logs("Scenario 1 (Scratch)", logs_sim_1)
    
    logs_sim_2 = exp_sim.run_scenario_2_full_transfer(episodes)
    evaluate_logs("Scenario 2 (Full)   ", logs_sim_2)
    
    logs_sim_3 = exp_sim.run_scenario_3_beta_transfer(beta_value, episodes)
    evaluate_logs(f"Scenario 3 (B={beta_value}) ", logs_sim_3)
    
    logs_sim_4 = exp_sim.run_scenario_4_selective_transfer(episodes)
    evaluate_logs("Scenario 4 (Select) ", logs_sim_4)

    # ==========================================
    # آزمایش دوم: محیط متفاوت
    # ==========================================
    print("\n" + "="*60)
    print("--- TRANSFER TO DIFFERENT ENVIRONMENT ---")
    exp_diff = TransferExperiment(base_env, different_env, source_q_table)
    
    logs_diff_1 = exp_diff.run_scenario_1_scratch(episodes)
    evaluate_logs("Scenario 1 (Scratch)", logs_diff_1)
    
    logs_diff_2 = exp_diff.run_scenario_2_full_transfer(episodes)
    evaluate_logs("Scenario 2 (Full)   ", logs_diff_2)
    
    logs_diff_3 = exp_diff.run_scenario_3_beta_transfer(beta_value, episodes)
    evaluate_logs(f"Scenario 3 (B={beta_value}) ", logs_diff_3)
    
    logs_diff_4 = exp_diff.run_scenario_4_selective_transfer(episodes)
    evaluate_logs("Scenario 4 (Select) ", logs_diff_4)

    print("\nTransfer Learning Experiments Completed Successfully!")

if __name__ == "__main__":
    main()