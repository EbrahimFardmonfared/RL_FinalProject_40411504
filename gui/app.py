import pygame
import sys
import os
import numpy as np

# اضافه کردن مسیر ریشه پروژه
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.maze import DynamicMazeEnv
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent
from gui.renderer import Renderer

class MazeApp:
    def __init__(self):
        pygame.init()
        self.fps = 10
        self.running = True
        self.paused = False
        self.show_policy = False
        
        self.env_type = 'Source'
        self.algo_name = 'Q-Learning'
        self.mode = 'Train'
        
        self.recent_successes = []
        self.episode = 1
        self.step_count = 0
        self.ep_reward = 0
        
        self.setup_env_and_agent()
        self.renderer = Renderer(self.env)
        
    def setup_env_and_agent(self):
        """راه‌اندازی محیط و عامل بر اساس انتخاب‌های کاربر در رابط کاربری"""
        self.env = DynamicMazeEnv(use_reward_shaping=False)
        if self.env_type == 'Similar' and hasattr(self.env, 'generate_similar_map'):
            self.env.generate_similar_map()
        elif self.env_type == 'Different' and hasattr(self.env, 'generate_different_map'):
            self.env.generate_different_map()
            
        if self.algo_name == 'Q-Learning':
            self.agent = QLearningAgent(self.env)
        elif self.algo_name == 'SARSA':
            self.agent = SarsaLambdaAgent(self.env)
        else:
            self.agent = ValueIterationAgent(self.env)
            print("Computing Value Iteration Optimal Policy... Please wait.")
            self.agent.run()
            print("Value Iteration Converged!")
            
        self.reset_episode()
        self.recent_successes = []
        self.episode = 1

    def reset_episode(self):
        self.state = self.env.reset()
        self.step_count = 0
        self.ep_reward = 0
        if hasattr(self.agent, 'E'):
            self.agent.E.clear()
        self.current_action = self.get_action_for_state(self.state)

    def get_epsilon(self):
        if self.mode == 'Eval' or self.algo_name == 'Value Iteration':
            return 0.0
        return getattr(self.agent, 'epsilon', 0.0)

    def get_action_for_state(self, state):
        if hasattr(self.agent, 'choose_action'):
            return self.agent.choose_action(state, self.get_epsilon())
        return self.agent.get_action(state)

    def handle_events(self):
        """مدیریت کلیدهای میانبر"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_p:
                    self.show_policy = not self.show_policy
                elif event.key == pygame.K_m:
                    self.mode = 'Eval' if self.mode == 'Train' else 'Train'
                    print(f"Mode changed to: {self.mode}")
                elif event.key == pygame.K_UP:
                    self.fps = min(120, self.fps + 5)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 5)
                elif event.key == pygame.K_1:
                    print("Environment changed to: Source"); self.env_type = 'Source'; self.setup_env_and_agent()
                elif event.key == pygame.K_2:
                    print("Environment changed to: Similar"); self.env_type = 'Similar'; self.setup_env_and_agent()
                elif event.key == pygame.K_3:
                    print("Environment changed to: Different"); self.env_type = 'Different'; self.setup_env_and_agent()
                elif event.key == pygame.K_q:
                    print("Algorithm changed to: Q-Learning"); self.algo_name = 'Q-Learning'; self.setup_env_and_agent()
                elif event.key == pygame.K_s:
                    print("Algorithm changed to: SARSA"); self.algo_name = 'SARSA'; self.setup_env_and_agent()
                elif event.key == pygame.K_v:
                    print("Algorithm changed to: Value Iteration"); self.algo_name = 'Value Iteration'; self.setup_env_and_agent()

    def step(self):
        if self.paused:
            return

        action = self.current_action
        next_state, reward, done, info = self.env.step(action)
        self.step_count += 1
        self.ep_reward += reward
        
        if self.mode == 'Train':
            if self.algo_name == 'Q-Learning':
                best_next = np.argmax([self.agent.get_q(next_state, a) for a in range(4)])
                td_target = reward + self.agent.gamma * self.agent.get_q(next_state, best_next)
                td_error = td_target - self.agent.get_q(self.state, action)
                self.agent.Q[(self.state, action)] += self.agent.alpha * td_error
                
            elif self.algo_name == 'SARSA':
                next_action = self.get_action_for_state(next_state)
                td_target = reward + self.agent.gamma * self.agent.get_q(next_state, next_action)
                td_error = td_target - self.agent.get_q(self.state, action)
                
                if getattr(self.agent, 'trace_type', 'replacing') == 'accumulating':
                    self.agent.E[(self.state, action)] = self.agent.E.get((self.state, action), 0.0) + 1.0
                else:
                    self.agent.E[(self.state, action)] = 1.0
                    
                for (s, a) in list(self.agent.E.keys()):
                    self.agent.Q[(s, a)] += self.agent.alpha * td_error * self.agent.E[(s, a)]
                    self.agent.E[(s, a)] *= self.agent.gamma * self.agent.lmbda
                    if self.agent.E[(s, a)] < 1e-4:
                        del self.agent.E[(s, a)]
                self.current_action = next_action

        if done or self.step_count >= 500:
            is_success = 1 if self.step_count < 500 and reward > 0 else 0
            self.recent_successes.append(is_success)
            if len(self.recent_successes) > 100:
                self.recent_successes.pop(0)
            
            if self.mode == 'Train' and hasattr(self.agent, 'decay_type'):
                if self.agent.decay_type == 'exponential':
                    self.agent.epsilon = max(self.agent.epsilon_end, self.agent.epsilon * self.agent.decay_rate)
                else:
                    self.agent.epsilon = max(self.agent.epsilon_end, self.agent.epsilon - self.agent.decay_rate)
            
            self.episode += 1
            self.reset_episode()
        else:
            self.state = next_state
            if self.algo_name == 'Q-Learning' or self.mode == 'Eval' or self.algo_name == 'Value Iteration':
                self.current_action = self.get_action_for_state(self.state)

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            self.handle_events()
            self.step()
            
            success_rate = (sum(self.recent_successes) / len(self.recent_successes) * 100) if self.recent_successes else 0.0
            
            stats = {
                'Mode': self.mode,
                'Algorithm': self.algo_name,
                'Environment': self.env_type,
                'Episode': self.episode,
                'Step': self.step_count,
                'Reward': round(self.ep_reward, 2),
                'Epsilon': round(self.get_epsilon(), 3),
                'Success Rate': f"{success_rate:.1f}%",
                'Has Key': 'Yes' if self.env.has_key else 'No',
                'FPS': self.fps
            }
            
            self.renderer.render(self.env, self.agent, stats, self.show_policy)
            clock.tick(self.fps)
        pygame.quit()

if __name__ == '__main__':
    app = MazeApp()
    app.run()