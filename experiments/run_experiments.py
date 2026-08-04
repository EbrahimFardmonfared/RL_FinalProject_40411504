import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from transfer.transfer_learning import TransferLearningExperiment

def run_all():
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/raw_data', exist_ok=True)

    print("--- Starting Transfer Learning Experiments ---")
    
    # 1. تنظیم محیط‌ها (محیط مقصد کمی سخت‌تر است)
    source_env = DynamicMazeEnv(grid_size=10, obstacle_speed=1, use_reward_shaping=False)
    target_env = DynamicMazeEnv(grid_size=10, obstacle_speed=2, use_reward_shaping=False)

    transfer_exp = TransferLearningExperiment(source_env, target_env)

    # 2. آموزش روی محیط مبدأ
    print("Training on Source Environment...")
    src_rewards, src_steps, _, _ = transfer_exp.train_source(episodes=500)

    # 3. آموزش روی محیط مقصد (از صفر - بدون دانش قبلی)
    print("Training on Target Environment (From Scratch)...")
    scratch_rewards, scratch_steps, _, _ = transfer_exp.train_target_from_scratch(episodes=500)

    # 4. آموزش روی محیط مقصد (با انتقال یادگیری)
    print("Training on Target Environment (With Transfer)...")
    trans_rewards, trans_steps, _, _ = transfer_exp.train_target_with_transfer(episodes=500)

    # رفع باگ: محاسبه عملکرد اولیه (Initial Performance) از روی آرایه rewards
    init_perf_scratch = sum(scratch_rewards[:10]) / 10 if len(scratch_rewards) >= 10 else 0
    init_perf_transfer = sum(trans_rewards[:10]) / 10 if len(trans_rewards) >= 10 else 0
    print(f"Initial Performance (First 10 episodes) -> Scratch: {init_perf_scratch:.2f} | Transfer: {init_perf_transfer:.2f}")

    # 5. ذخیره نتایج در فایل CSV
    df = pd.DataFrame({
        'Episode': range(1, 501),
        'Scratch_Rewards': scratch_rewards,
        'Transfer_Rewards': trans_rewards
    })
    df.to_csv('results/raw_data/transfer_learning_results.csv', index=False)

    # 6. رسم نمودار مقایسه انتقال یادگیری
    window = 20
    plt.figure(figsize=(10, 6))
    plt.plot(pd.Series(scratch_rewards).rolling(window).mean(), label='From Scratch', color='red')
    plt.plot(pd.Series(trans_rewards).rolling(window).mean(), label='With Transfer', color='green')
    plt.title(f'Transfer Learning Performance (Moving Average {window})')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/figures/transfer_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ Transfer Learning experiments completed successfully.")

if __name__ == "__main__":
    run_all()