import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from transfer.transfer_learning import TransferLearningExperiment

def calculate_success(steps_list, max_steps=500):
    # اگر تعداد قدم‌ها کمتر از سقف باشد، یعنی با موفقیت به هدف رسیده است
    return [1 if s < max_steps else 0 for s in steps_list]

def save_and_plot_results(target_name, results_dict):
    # ذخیره ۶ فایل CSV برای هر محیط
    for method_name, data in results_dict.items():
        rewards, steps, wall_hits, penalty_hits = data
        df = pd.DataFrame({
            'Episode': range(1, len(rewards) + 1),
            'Rewards': rewards,
            'Steps': steps,
            'Wall_Hits': wall_hits,
            'Penalty_Hits': penalty_hits,
            'Success': calculate_success(steps)
        })
        safe_method_name = method_name.replace('=', '').replace('.', '_')
        df.to_csv(f'results/raw_data/transfer_{target_name}_{safe_method_name}.csv', index=False)
        
    # رسم نمودار Transfer Curves هماهنگ با داده‌های جدید
    window = 20
    plt.figure(figsize=(10, 6))
    for method_name, data in results_dict.items():
        rewards = data[0]
        plt.plot(pd.Series(rewards).rolling(window).mean(), label=method_name)
        
    plt.title(f'Transfer Learning Performance ({target_name} Environment)')
    plt.xlabel('Episodes')
    plt.ylabel(f'Total Reward (Moving Average {window})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/figures/Transfer_Curves_{target_name}.png', dpi=300)
    plt.close()

def run_all():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/raw_data', exist_ok=True)

    print("--- Starting Full Transfer Learning Experiments ---")
    
    # 1. ساخت محیط مبدأ (بدون پارامترهای غلط)
    source_env = DynamicMazeEnv(use_reward_shaping=False)
    
    # 2. ساخت محیط مشابه
    similar_env = DynamicMazeEnv(use_reward_shaping=False)
    if hasattr(similar_env, 'generate_similar_map'):
        similar_env.generate_similar_map()
        
    # 3. ساخت محیط متفاوت
    different_env = DynamicMazeEnv(use_reward_shaping=False)
    if hasattr(different_env, 'generate_different_map'):
        different_env.generate_different_map()

    episodes = 500

    # اجرای سناریوها برای هر دو محیط
    for target_name, target_env in [("Similar", similar_env), ("Different", different_env)]:
        print(f"\n==================================================")
        print(f"Evaluating Transfer on {target_name} Environment")
        print(f"==================================================")
        
        exp = TransferLearningExperiment(source_env, target_env)
        
        print("1. Training Source Agent (Building Knowledge)...")
        exp.train_source(episodes)
        
        results = {}
        
        print("2. Target: From Scratch (No Transfer)...")
        results['Scratch'] = exp.train_target_from_scratch(episodes)
        
        print("3. Target: Full Transfer...")
        results['Full'] = exp.train_target_full_transfer(episodes)
        
        for beta in [0.25, 0.5, 0.75]:
            print(f"4. Target: Beta Transfer (Beta={beta})...")
            results[f'Beta_{beta}'] = exp.train_target_beta_transfer(episodes, beta=beta)
            
        print("5. Target: Selective Transfer...")
        results['Selective'] = exp.train_target_selective_transfer(episodes)
        
        save_and_plot_results(target_name, results)
        print(f"✅ Data and Plots saved for {target_name} Environment.")

    print("\n🎉 Transfer Learning benchmarks completed! 12 CSVs and 2 PNGs generated.")

if __name__ == "__main__":
    run_all()