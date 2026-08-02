import sys
import os
import pygame
import time

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from gui.renderer import MazeRenderer

def extract_policy(q_table):
    """استخراج بهترین عمل برای هر حالت جهت رسم فلش‌های سیاست"""
    policy = {}
    for (state, action), q_val in q_table.items():
        if state not in policy or q_val > q_table.get((state, policy.get(state)), -float('inf')):
            policy[state] = action
    return policy

def main():
    print("Initializing GUI and Training Agent in background (Please wait)...")
    env = DynamicMazeEnv()
    
    # آموزش سریع یک عامل برای نمایش
    agent = QLearningAgent(env, decay_type='exponential')
    agent.train(episodes=400)
    policy_dict = extract_policy(agent.Q)
    
    renderer = MazeRenderer(env)
    
    running = True
    paused = False
    show_policy = False
    delay = 0.15  # سرعت اولیه انیمیشن
    
    episode = 1
    state = env.reset()
    step = 0
    total_reward = 0
    status = "Running"
    
    while running:
        # مدیریت کلیدهای کنترلی کیبورد
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    status = "Paused" if paused else "Running"
                elif event.key == pygame.K_r:
                    state = env.reset()
                    step = 0
                    total_reward = 0
                    status = "Running"
                elif event.key == pygame.K_p:
                    show_policy = not show_policy
                elif event.key == pygame.K_UP:
                    delay = max(0.01, delay - 0.05) # افزایش سرعت (کاهش تاخیر)
                elif event.key == pygame.K_DOWN:
                    delay = min(0.5, delay + 0.05)  # کاهش سرعت
        
        # اجرای منطق بازی اگر متوقف نشده باشد
        if not paused and status == "Running":
            q_values = [agent.get_q(state, a) for a in range(4)]
            # استفاده از سیاست حریصانه (بدون اکتشاف) برای ارزیابی نهایی
            best_actions = [a for a in range(4) if q_values[a] == max(q_values)]
            action = best_actions[0] if best_actions else 0
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            step += 1
            total_reward += reward
            
            if done:
                status = "Goal Reached!" if reward > 0 else "Failed"
                
        # بروزرسانی و رندر تصویر
        info = {
            'episode': episode,
            'step': step,
            'reward': total_reward,
            'key': state[2],
            'status': status
        }
        
        renderer.draw_state(state, info, policy_dict if show_policy else None)
        
        # کنترل سرعت فریم‌ها
        time.sleep(delay)
        
        # بازنشانی خودکار اپیزود در صورت اتمام
        if status != "Running" and not paused:
            time.sleep(1.5)
            episode += 1
            state = env.reset()
            step = 0
            total_reward = 0
            status = "Running"

    pygame.quit()

if __name__ == "__main__":
    main()