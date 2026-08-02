import pygame
import numpy as np

class MazeRenderer:
    def __init__(self, env, cell_size=40):
        self.env = env
        self.cell_size = cell_size
        self.width = env.grid_size * cell_size
        self.height = env.grid_size * cell_size + 120  # فضای اضافه برای پنل اطلاعات
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dynamic Maze RL Agent")
        
        # تعریف رنگ‌های متمایز برای محیط طبق داکیومنت
        self.COLORS = {
            'empty': (240, 248, 255),    # آبی بسیار روشن
            'wall': (47, 79, 79),        # خاکستری تیره
            'penalty': (220, 20, 60),    # قرمز
            'key': (255, 215, 0),        # طلایی
            'door': (139, 69, 19),       # قهوه‌ای
            'goal': (50, 205, 50),       # سبز
            'agent': (30, 144, 255),     # آبی روشن (عامل)
            'text': (0, 0, 0),           # مشکی
            'bg': (200, 200, 200)        # پس‌زمینه پنل
        }
        self.font = pygame.font.SysFont('Tahoma', 16, bold=True)

    def draw_state(self, state, info_dict, policy=None):
        self.screen.fill(self.COLORS['bg'])
        agent_r, agent_c = state[0], state[1]
        has_key = state[2]
        
        # رسم شبکه و اجزای نقشه
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
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 1) # خطوط شبکه

                # رسم فلش‌های سیاست در صورت فعال بودن
                if policy and cell_type not in [self.env.WALL, self.env.GOAL]:
                    best_a = policy.get(((r, c, has_key)), None)
                    if best_a is not None:
                        self._draw_arrow(r, c, best_a)

        # رسم عامل (دایره آبی)
        agent_rect = pygame.Rect(agent_c * self.cell_size + 5, agent_r * self.cell_size + 5, self.cell_size - 10, self.cell_size - 10)
        pygame.draw.ellipse(self.screen, self.COLORS['agent'], agent_rect)
        
        # رسم پنل اطلاعات لحظه‌ای
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
        y_offset = self.env.grid_size * self.cell_size + 10
        x_offset = 15
        
        texts = [
            f"Episode: {info_dict.get('episode', 0)}",
            f"Step: {info_dict.get('step', 0)}",
            f"Reward: {info_dict.get('reward', 0)}",
            f"Key: {'Obtained' if info_dict.get('key', 0) else 'Missing'}",
            f"Status: {info_dict.get('status', 'Running')}",
            "Keys: [Space] Pause | [R] Reset | [P] Policy | [Up/Down] Speed"
        ]
        
        for idx, text in enumerate(texts):
            col = idx % 3
            row = idx // 3
            img = self.font.render(text, True, self.COLORS['text'])
            self.screen.blit(img, (x_offset + col * 180, y_offset + row * 30))