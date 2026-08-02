import pygame
import numpy as np

class MazeRenderer:
    def __init__(self, env, cell_size=30): 
        self.env = env
        self.cell_size = cell_size
        self.width = env.grid_size * cell_size
        # حداقل عرض پنجره را مشخص می‌کنیم تا نوشته‌ها جا شوند
        self.window_width = max(self.width, 450)
        # ارتفاع را بیشتر کردیم تا اطلاعات ستونی و کامل نوشته شوند
        self.height = env.grid_size * cell_size + 170 
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.height))
        pygame.display.set_caption("Dynamic Maze RL Agent")
        
        self.COLORS = {
            'empty': (240, 248, 255),    
            'wall': (47, 79, 79),        
            'penalty': (220, 20, 60),    
            'key': (255, 215, 0),        
            'door': (139, 69, 19),       
            'goal': (50, 205, 50),       
            'agent': (30, 144, 255),     
            'patrol': (128, 0, 128),     
            'text': (0, 0, 0),           
            'bg': (220, 220, 220)        
        }
        self.font = pygame.font.SysFont('Tahoma', 14, bold=True)
        self.small_font = pygame.font.SysFont('Tahoma', 12, bold=True)

    def draw_state(self, state, info_dict, policy=None):
        self.screen.fill(self.COLORS['bg'])
        agent_r, agent_c = state[0], state[1]
        has_key = state[2]
        
        # حاشیه برای وسط‌چین کردن نقشه در صورت عریض بودن پنجره
        x_offset_map = (self.window_width - self.width) // 2
        
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                rect = pygame.Rect(x_offset_map + c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                cell_type = self.env.grid[r, c]
                
                color = self.COLORS['empty']
                if cell_type == self.env.WALL:
                    color = self.COLORS['wall']
                elif cell_type == self.env.PENALTY:
                    color = self.COLORS['penalty']
                elif cell_type == self.env.KEY and not has_key:
                    color = self.COLORS['key']
                elif cell_type == self.env.DOOR:
                    color = self.COLORS['empty'] if has_key else self.COLORS['door']
                elif cell_type == self.env.GOAL:
                    color = self.COLORS['goal']
                    
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)

                if policy and cell_type not in [self.env.WALL, self.env.GOAL]:
                    best_a = policy.get(((r, c, has_key)), None)
                    if best_a is not None:
                        self._draw_arrow(r, c, best_a, x_offset_map)

        if hasattr(self.env, 'patrol_route') and self.env.patrol_route:
            step = info_dict.get('step', 0)
            route_len = len(self.env.patrol_route)
            
            if route_len > 1:
                idx = step % ((route_len - 1) * 2)
                if idx >= route_len:
                    idx = (route_len - 1) * 2 - idx
            else:
                idx = 0
                
            pr, pc = self.env.patrol_route[idx]
            patrol_rect = pygame.Rect(x_offset_map + pc * self.cell_size + 4, pr * self.cell_size + 4, self.cell_size - 8, self.cell_size - 8)
            pygame.draw.rect(self.screen, self.COLORS['patrol'], patrol_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), patrol_rect, 1)

        agent_rect = pygame.Rect(x_offset_map + agent_c * self.cell_size + 3, agent_r * self.cell_size + 3, self.cell_size - 6, self.cell_size - 6)
        pygame.draw.ellipse(self.screen, self.COLORS['agent'], agent_rect)
        pygame.draw.ellipse(self.screen, (0, 0, 0), agent_rect, 1) 
        
        self._draw_info(info_dict)
        pygame.display.flip()

    def _draw_arrow(self, r, c, action, x_offset_map):
        center = (x_offset_map + c * self.cell_size + self.cell_size//2, r * self.cell_size + self.cell_size//2)
        color = (100, 100, 100)
        offset = self.cell_size // 3
        if action == 0:   
            end = (center[0], center[1] - offset)
        elif action == 1: 
            end = (center[0], center[1] + offset)
        elif action == 2: 
            end = (center[0] - offset, center[1])
        else:             
            end = (center[0] + offset, center[1])
        pygame.draw.line(self.screen, color, center, end, 2)
        pygame.draw.circle(self.screen, color, end, 2)

    def _draw_info(self, info_dict):
        y_base = self.env.grid_size * self.cell_size + 15
        x_left = 20
        x_mid = 220
        
        # چیدمان ستونی برای استفاده از کلمات کامل
        ep_img = self.font.render(f"Episode: {info_dict.get('episode', 0)}", True, self.COLORS['text'])
        step_img = self.font.render(f"Step: {info_dict.get('step', 0)}", True, self.COLORS['text'])
        self.screen.blit(ep_img, (x_left, y_base))
        self.screen.blit(step_img, (x_mid, y_base))
        
        rew_img = self.font.render(f"Reward: {info_dict.get('reward', 0)}", True, self.COLORS['text'])
        key_status = 'Yes' if info_dict.get('key', 0) else 'No'
        key_img = self.font.render(f"Key: {key_status}", True, self.COLORS['text'])
        self.screen.blit(rew_img, (x_left, y_base + 25))
        self.screen.blit(key_img, (x_mid, y_base + 25))
        
        stat_img = self.font.render(f"Status: {info_dict.get('status', 'Running')}", True, self.COLORS['text'])
        self.screen.blit(stat_img, (x_left, y_base + 50))
        
        algo_img = self.font.render(f"Active Policy: {info_dict.get('algorithm', 'Q-Learning')}", True, (0, 100, 200))
        self.screen.blit(algo_img, (x_left, y_base + 75))
        
        # راهنمای کامل دکمه‌ها در دو خط
        ctrl_text1 = "[Space]: Pause  |  [R]: Reset  |  [P]: Show Policy"
        ctrl_text2 = "[A]: Switch Algorithm  |  [Up/Down]: Speed Control"
        ctrl_img1 = self.small_font.render(ctrl_text1, True, (80, 80, 80))
        ctrl_img2 = self.small_font.render(ctrl_text2, True, (80, 80, 80))
        self.screen.blit(ctrl_img1, (x_left, y_base + 110))
        self.screen.blit(ctrl_img2, (x_left, y_base + 130))