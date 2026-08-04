import pygame
import numpy as np

class Renderer:
    def __init__(self, env, cell_size=30):
        self.cell_size = cell_size
        self.grid_size = env.grid_size
        self.width = self.grid_size * self.cell_size
        self.dashboard_height = 200
        self.height = self.width + self.dashboard_height
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Dynamic Maze Dashboard")
        self.font = pygame.font.SysFont('Arial', 16, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 13)
        self.key_font = pygame.font.SysFont('Arial', 14, bold=True)
        
        self.colors = {
            'empty': (240, 240, 240),
            'wall': (40, 40, 40),
            'penalty': (200, 50, 50),
            'agent': (50, 150, 250),
            'door_locked': (139, 69, 19),
            'door_open': (160, 160, 160),
            'goal': (50, 200, 50),
            'obstacle': (255, 100, 0),
            'bg': (20, 20, 30),
            'text': (220, 220, 220)
        }

    def draw_grid(self, env):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = env.grid[r, c]
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                
                color = self.colors['empty']
                if cell == env.WALL: color = self.colors['wall']
                elif cell == env.PENALTY: color = self.colors['penalty']
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                
        # هدف
        goal_rect = pygame.Rect(env.goal_pos[1]*self.cell_size, env.goal_pos[0]*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.colors['goal'], goal_rect)
        
        # کلید: حرف K درون دایره با پس‌زمینه زرد
        if not env.has_key:
            kr, kc = env.key_pos
            cell_rect = pygame.Rect(kc * self.cell_size, kr * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 215, 0), cell_rect)
            pygame.draw.rect(self.screen, (200, 200, 200), cell_rect, 1)
            
            center_x = kc * self.cell_size + self.cell_size // 2
            center_y = kr * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 3
            pygame.draw.circle(self.screen, (220, 180, 0), (center_x, center_y), radius)
            pygame.draw.circle(self.screen, (100, 80, 0), (center_x, center_y), radius, 1)
            
            text_surf = self.key_font.render("K", True, (20, 20, 20))
            text_rect = text_surf.get_rect(center=(center_x, center_y))
            self.screen.blit(text_surf, text_rect)
            
        # در
        door_color = self.colors['door_open'] if env.has_key else self.colors['door_locked']
        door_rect = pygame.Rect(env.door_pos[1]*self.cell_size, env.door_pos[0]*self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, door_color, door_rect)
        
        # مانع متحرک
        obs_pos = env.patrol_route[env.patrol_idx]
        obs_rect = pygame.Rect(obs_pos[1]*self.cell_size+4, obs_pos[0]*self.cell_size+4, self.cell_size-8, self.cell_size-8)
        pygame.draw.rect(self.screen, self.colors['obstacle'], obs_rect)
        
        # عامل
        agent_rect = pygame.Rect(env.agent_pos[1]*self.cell_size+5, env.agent_pos[0]*self.cell_size+5, self.cell_size-10, self.cell_size-10)
        pygame.draw.ellipse(self.screen, self.colors['agent'], agent_rect)

    def draw_arrow(self, cx, cy, direction, length):
        color = (0, 0, 0)
        if direction == 0: # بالا
            end = (cx, cy - length)
            pygame.draw.line(self.screen, color, (cx, cy), end, 2)
            pygame.draw.polygon(self.screen, color, [(cx, cy - length - 3), (cx - 4, cy - length + 4), (cx + 4, cy - length + 4)])
        elif direction == 1: # راست
            end = (cx + length, cy)
            pygame.draw.line(self.screen, color, (cx, cy), end, 2)
            pygame.draw.polygon(self.screen, color, [(cx + length + 3, cy), (cx + length - 4, cy - 4), (cx + length - 4, cy + 4)])
        elif direction == 2: # پایین
            end = (cx, cy + length)
            pygame.draw.line(self.screen, color, (cx, cy), end, 2)
            pygame.draw.polygon(self.screen, color, [(cx, cy + length + 3), (cx - 4, cy + length - 4), (cx + 4, cy + length - 4)])
        elif direction == 3: # چپ
            end = (cx - length, cy)
            pygame.draw.line(self.screen, color, (cx, cy), end, 2)
            pygame.draw.polygon(self.screen, color, [(cx - length - 3, cy), (cx - length + 4, cy - 4), (cx - length + 4, cy + 4)])

    def draw_policy(self, env, agent):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if env.grid[r, c] == env.WALL:
                    continue
                state = (r, c, int(env.has_key), env.patrol_idx)
                
                if hasattr(agent, 'get_q'):
                    q_vals = [agent.get_q(state, a) for a in range(4)]
                    if all(q == 0 for q in q_vals): continue
                    best_a = np.argmax(q_vals)
                elif hasattr(agent, 'get_action'):
                    best_a = agent.get_action(state)
                else:
                    continue
                    
                cx = c * self.cell_size + self.cell_size//2
                cy = r * self.cell_size + self.cell_size//2
                length = self.cell_size // 3
                
                self.draw_arrow(cx, cy, best_a, length)

    def draw_dashboard(self, stats):
        dash_rect = pygame.Rect(0, self.width, self.width, self.dashboard_height)
        pygame.draw.rect(self.screen, self.colors['bg'], dash_rect)
        
        header = self.font.render("RL Agent Interactive Dashboard", True, (255, 215, 0))
        self.screen.blit(header, (15, self.width + 5))
        
        col1_x, col2_x, y_start, y_gap = 15, self.width // 2, self.width + 35, 20
        
        left_stats = [
            f"Algorithm: {stats['Algorithm']}",
            f"Mode: {stats['Mode']}",
            f"Environment: {stats['Environment']}",
            f"Episode: {stats['Episode']}",
            f"Step: {stats['Step']}"
        ]
        right_stats = [
            f"Epsilon: {stats['Epsilon']}",
            f"Success Rate: {stats['Success Rate']}",
            f"Current Reward: {stats['Reward']}",
            f"Has Key: {stats['Has Key']}",
            f"Speed (FPS): {stats['FPS']}"
        ]
        
        for i, text in enumerate(left_stats):
            img = self.font.render(text, True, self.colors['text'])
            self.screen.blit(img, (col1_x, y_start + i * y_gap))
            
        for i, text in enumerate(right_stats):
            img = self.font.render(text, True, self.colors['text'])
            self.screen.blit(img, (col2_x, y_start + i * y_gap))
            
        c1 = "[SPACE]: Pause  |  [M]: Train/Eval  |  [P]: Show Policy  |  [+/-]: Speed"
        c2 = "Env: [1] Source, [2] Similar, [3] Different"
        c3 = "Algo: [Q] Q-Learning, [S] SARSA, [V] Value Iteration"
        
        self.screen.blit(self.small_font.render(c1, True, (180, 180, 180)), (10, self.height - 60))
        self.screen.blit(self.small_font.render(c2, True, (150, 200, 250)), (10, self.height - 40))
        self.screen.blit(self.small_font.render(c3, True, (250, 200, 150)), (10, self.height - 20))

    def render(self, env, agent, stats, show_policy):
        self.screen.fill((0, 0, 0))
        self.draw_grid(env)
        if show_policy:
            self.draw_policy(env, agent)
        self.draw_dashboard(stats)
        pygame.display.flip()