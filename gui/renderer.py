import pygame
import numpy as np

class MazeRenderer:
    def __init__(self, env, cell_size=30): 
        self.env = env
        self.cell_size = cell_size
        self.width = env.grid_size * cell_size
        self.window_width = max(self.width, 450)
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

    def draw_state(self, state, info=None, policy_map=None):
        """رسم کامل وضعیت فعلی محیط شامل نقشه، مانع متحرک، عامل و پنل اطلاعات/فلش‌های سیاست"""
        self.screen.fill((255, 255, 255)) # پاک کردن صفحه برای جلوگیری از صفحه سیاه
        
        # 1. رسم شبکه‌ی اصلی (دیوارها، جریمه‌ها، کلید، در و هدف)
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                cell_rect = pygame.Rect(
                    c * self.cell_size, 
                    r * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                cell_type = self.env.grid[r, c]
                
                if cell_type == self.env.WALL:
                    color = (50, 50, 50)
                elif cell_type == self.env.PENALTY:
                    color = (255, 100, 100)
                elif cell_type == self.env.KEY and not self.env.has_key:
                    color = (255, 215, 0)
                elif cell_type == self.env.DOOR:
                    color = (139, 69, 19) if not self.env.has_key else (210, 180, 140)
                elif cell_type == self.env.GOAL:
                    color = (50, 205, 50)
                else:
                    color = (240, 240, 240)
                    
                pygame.draw.rect(self.screen, color, cell_rect)
                pygame.draw.rect(self.screen, (200, 200, 200), cell_rect, 1)

        # 2. رسم فلش‌های سیاست (اگر فعال باشد)
        if policy_map:
            for (r, c, has_key), action in policy_map.items():
                if self.env.grid[r, c] != self.env.WALL:
                    self._draw_arrow(r, c, action)

        # 3. همگام‌سازی مانع متحرک
        if len(state) == 4:
            patrol_idx = state[3]
            pr, pc = self.env.patrol_route[patrol_idx]
            patrol_rect = pygame.Rect(
                pc * self.cell_size + 4, 
                pr * self.cell_size + 4, 
                self.cell_size - 8, 
                self.cell_size - 8
            )
            pygame.draw.rect(self.screen, (128, 0, 128), patrol_rect)

        # 4. رسم موقعیت عامل
        ar, ac = state[0], state[1]
        agent_center = (
            ac * self.cell_size + self.cell_size // 2, 
            ar * self.cell_size + self.cell_size // 2
        )
        pygame.draw.circle(self.screen, (0, 102, 204), agent_center, self.cell_size // 3)

        # 5. به‌روزرسانی نهایی صفحه (حیاتی برای جلوگیری از صفحه سیاه)
        pygame.display.flip()

    def _draw_arrow(self, r, c, action, x_offset_map=0):
        # پیدا کردن مرکز هر خانه در رابط گرافیکی
        center = (x_offset_map + c * self.cell_size + self.cell_size//2, 
                  r * self.cell_size + self.cell_size//2)
        color = (100, 100, 100) # رنگ خاکستری برای فلش‌ها
        offset = self.cell_size // 3
        
        # نگاشت صحیح بر اساس موتور محیط: 
        # 0: بالا، 1: راست، 2: پایین، 3: چپ
        if action == 0:   
            end = (center[0], center[1] - offset)
        elif action == 1: 
            end = (center[0] + offset, center[1])
        elif action == 2: 
            end = (center[0], center[1] + offset)
        elif action == 3: 
            end = (center[0] - offset, center[1])
        
        pygame.draw.line(self.screen, color, center, end, 2)
        pygame.draw.circle(self.screen, color, end, 2)
        
    def _draw_info(self, info_dict):
        y_base = self.env.grid_size * self.cell_size + 15
        x_left = 20
        x_mid = 220
        
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
        
        ctrl_text1 = "[Space]: Pause  |  [R]: Reset  |  [P]: Show Policy"
        ctrl_text2 = "[A]: Switch Algorithm  |  [Up/Down]: Speed Control"
        ctrl_img1 = self.small_font.render(ctrl_text1, True, (80, 80, 80))
        ctrl_img2 = self.small_font.render(ctrl_text2, True, (80, 80, 80))
        self.screen.blit(ctrl_img1, (x_left, y_base + 110))
        self.screen.blit(ctrl_img2, (x_left, y_base + 130))