import sys
import os
import pygame
import time

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent
from gui.renderer import MazeRenderer

def get_agent_policy(env, agent, algo_name):
    """استخراج امن سیاست برای نمایش بصری روی نقشه (با لحاظ کردن فضای حالت ۴بعدی)"""
    policy = {}
    default_p = 0 # استفاده از برش پایه مانع متحرک برای نمایش دوبعدی سیاست
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            if env.grid[r, c] == env.WALL:
                continue
            for has_key in [0, 1]:
                state = (r, c, has_key, default_p)
                if algo_name == 'Value Iteration':
                    if hasattr(agent, 'get_action'):
                        policy[(r, c, has_key)] = agent.get_action(state)
                    elif hasattr(agent, 'policy') and isinstance(agent.policy, dict):
                        policy[(r, c, has_key)] = agent.policy.get(state, 0)
                else:
                    q_values = [agent.get_q(state, a) for a in range(4)]
                    if any(q != 0 for q in q_values):
                        best_a = q_values.index(max(q_values))
                        policy[(r, c, has_key)] = best_a
    return policy

def get_action_safely(agent, algo_name, state):
    """دریافت امن عمل برای جلوگیری از خطای عامل"""
    if algo_name == 'Value Iteration':
        if hasattr(agent, 'get_action'):
            return agent.get_action(state)
        elif hasattr(agent, 'policy') and isinstance(agent.policy, dict):
            return agent.policy.get(state, 0)
        return 0 
    else:
        q_values = [agent.get_q(state, a) for a in range(4)]
        best_actions = [a for a in range(4) if q_values[a] == max(q_values)]
        return best_actions[0] if best_actions else 0

def main():
    env = DynamicMazeEnv()
    renderer = MazeRenderer(env, cell_size=30)
    
    state = env.reset()
    loading_info = {
        'episode': 0, 'step': 0, 'reward': 0, 'key': 0,
        'status': "Training All Agents... Please wait!",
        'algorithm': "Initializing..."
    }
    renderer.draw_state(state, loading_info, None)
    pygame.event.pump() 
    
    print("Training Value Iteration Agent...")
    vi_agent = ValueIterationAgent(env, gamma=0.9, theta=1e-6)
    trained_vi = False
    for method in ['train', 'solve', 'value_iteration', 'run', 'optimize']:
        if hasattr(vi_agent, method):
            getattr(vi_agent, method)()
            trained_vi = True
            break
            
    print("Training Q-Learning Agent...")
    ql_agent = QLearningAgent(env, decay_type='exponential')
    ql_agent.train(episodes=400)
    
    print("Training SARSA Agent...")
    sarsa_agent = SarsaLambdaAgent(env, lmbda=0.9, trace_type='replacing')
    sarsa_agent.train(episodes=400)
    
    algo_names = ['Q-Learning', 'SARSA(lambda=0.9)', 'Value Iteration']
    agents_dict = {
        'Q-Learning': ql_agent,
        'SARSA(lambda=0.9)': sarsa_agent,
        'Value Iteration': vi_agent
    }
    
    current_idx = 0
    current_algo_name = algo_names[current_idx]
    current_agent = agents_dict[current_algo_name]
    policy_dict = get_agent_policy(env, current_agent, current_algo_name)
    
    running = True
    paused = False
    show_policy = False
    delay = 0.15 
    
    episode = 1
    state = env.reset()
    step = 0
    total_reward = 0
    status = "Running"
    
    print("Training Complete! Starting visualization.")
    
    while running:
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
                elif event.key == pygame.K_a: 
                    current_idx = (current_idx + 1) % 3
                    current_algo_name = algo_names[current_idx]
                    current_agent = agents_dict[current_algo_name]
                    policy_dict = get_agent_policy(env, current_agent, current_algo_name)
                    state = env.reset()
                    step = 0
                    total_reward = 0
                    status = "Running"
                elif event.key == pygame.K_UP:
                    delay = max(0.01, delay - 0.05) 
                elif event.key == pygame.K_DOWN:
                    delay = min(0.5, delay + 0.05)  
        
        if not paused and status not in ["Paused", "Goal Reached!", "Failed"]:
            action = get_action_safely(current_agent, current_algo_name, state)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            step += 1
            total_reward += reward
            
            if done:
                status = "Goal Reached!" if reward > 0 else "Failed"
                
        info = {
            'episode': episode,
            'step': step,
            'reward': total_reward,
            'key': state[2],
            'status': status,
            'algorithm': current_algo_name
        }
        
        renderer.draw_state(state, info, policy_dict if show_policy else None)
        time.sleep(delay)
        
        if status in ["Goal Reached!", "Failed"] and not paused:
            time.sleep(1.5)
            episode += 1
            state = env.reset()
            step = 0
            total_reward = 0
            status = "Running"

    pygame.quit()

if __name__ == "__main__":
    main()