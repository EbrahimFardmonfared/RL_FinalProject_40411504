import pygame
import numpy as np

class MazeRenderer:
    def __init__(self, env, cell_size=45): # سایز خانه‌ها را بزرگتر کردیم
        self.env = env
        self.cell_size = cell_size
        self.width = env.grid_size * cell_size
        # فضای پایین صفحه را از 120 به 160 افزایش دادیم تا نوشته‌ها جا شوند
        self.height = env.grid_size * cell_size + 160 
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dynamic Maze RL Agent")
        
        self.COLORS = {
            'empty': (240, 248, 255),    
            'wall': (47, 79, 79),        
            'penalty': (220, 20, 60),    
            'key': (255, 215, 0),        
            'door': (139, 69, 19),       
            'goal': (50, 205, 50),       
            'agent': (30, 144, 255),     
            'patrol': (128, 0, 128),     # رنگ بنفش برای مانع متحرک
            'text': (0, 0, 0),           
            'bg': (220, 220, 220)        
        }
        self.font = pygame.font.SysFont('Tahoma', 16, bold=True)
        self.small_font = pygame.font.SysFont('Tahoma', 14, bold=True)

    def draw_state(self, state, info_dict, policy=None):
        self.screen.fill(self.COLORS['bg'])
        agent_r, agent_c = state[0], state[1]
        has_key = state[2]
        
        # رسم شبکه و اجزای ثابت نقشه
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
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
                        self._draw_arrow(r, c, best_a)

        # رسم مانع متحرک (مربع بنفش)
        if hasattr(self.env, 'patrol_route') and self.env.patrol_route:
            step = info_dict.get('step', 0)
            route_len = len(self.env.patrol_route)
            
            # محاسبه پینگ‌پونگی برای رفت و برگشت مانع در مسیر
            if route_len > 1:
                idx = step % ((route_len - 1) * 2)
                if idx >= route_len:
                    idx = (route_len - 1) * 2 - idx
            else:
                idx = 0
                
            pr, pc = self.env.patrol_route[idx]
            patrol_rect = pygame.Rect(pc * self.cell_size + 8, pr * self.cell_size + 8, self.cell_size - 16, self.cell_size - 16)
            pygame.draw.rect(self.screen, self.COLORS['patrol'], patrol_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), patrol_rect, 2)

        # رسم عامل (دایره آبی)
        agent_rect = pygame.Rect(agent_c * self.cell_size + 5, agent_r * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10)
        pygame.draw.ellipse(self.screen, self.COLORS['agent'], agent_rect)
        pygame.draw.ellipse(self.screen, (0, 0, 0), agent_rect, 2) # حاشیه دایره
        
        self._draw_info(info_dict)
        pygame.display.flip()

    def _draw_arrow(self, r, c, action):
        center = (c * self.cell_size + self.cell_size//2, r * self.cell_size + self.cell_size//2)
        color = (100, 100, 100)
        offset = self.cell_size // 3
        if action == 0:   # UP
            end = (center[0], center[1] - offset)
        elif action == 1: # DOWN
            end = (center[0], center[1] + offset)
        elif action == 2: # LEFT
            end = (center[0] - offset, center[1])
        else:             # RIGHT
            end = (center[0] + offset, center[1])
        pygame.draw.line(self.screen, color, center, end, 2)
        pygame.draw.circle(self.screen, color, end, 3)

    def _draw_info(self, info_dict):
        y_offset = self.env.grid_size * self.cell_size + 20
        x_offset = 20
        
        # سطر اول اطلاعات
        ep_img = self.font.render(f"Episode: {info_dict.get('episode', 0)}", True, self.COLORS['text'])
        step_img = self.font.render(f"Step: {info_dict.get('step', 0)}", True, self.COLORS['text'])
        rew_img = self.font.render(f"Reward: {info_dict.get('reward', 0)}", True, self.COLORS['text'])
        
        self.screen.blit(ep_img, (x_offset, y_offset))
        self.screen.blit(step_img, (x_offset + 160, y_offset))
        self.screen.blit(rew_img, (x_offset + 320, y_offset))
        
        # سطر دوم اطلاعات
        key_status = 'Obtained' if info_dict.get('key', 0) else 'Missing'
        key_img = self.font.render(f"Key: {key_status}", True, self.COLORS['text'])
        stat_img = self.font.render(f"Status: {info_dict.get('status', 'Running')}", True, self.COLORS['text'])
        
        self.screen.blit(key_img, (x_offset, y_offset + 40))
        self.screen.blit(stat_img, (x_offset + 160, y_offset + 40))
        
        # سطر سوم (راهنمای دکمه‌ها)
        controls_text = "Controls: [Space] Pause  |  [R] Reset  |  [P] Policy  |  [Up/Down] Speed"
        ctrl_img = self.small_font.render(controls_text, True, (80, 80, 80))
        self.screen.blit(ctrl_img, (x_offset, y_offset + 90))