import pygame
import sys

# پالت رنگی استاندارد
COLORS = {
    "background": (248, 249, 250),
    "wall": (45, 52, 54),
    "penalty": (235, 77, 75),
    "key": (241, 196, 15),
    "door_locked": (140, 85, 45),
    "door_open": (248, 249, 250),
    "goal": (46, 204, 113),
    "path": (255, 255, 255),
    "grid_line": (220, 224, 230),
    "obstacle": (142, 68, 173),
    "agent": (41, 128, 185),
    "arrow": (110, 110, 110),
    "panel_bg": (234, 238, 242),
    "panel_border": (189, 195, 199),
    "text_dark": (40, 40, 40),
    "text_gray": (70, 80, 90),
    "text_blue": (41, 128, 185),
    "text_green": (39, 174, 96),
    "text_red": (192, 57, 43)
}

class MazeRenderer:
    def __init__(self, env, cell_size=30):
        self.env = env
        self.cell_size = cell_size
        
        self.map_width = env.grid_size * cell_size
        self.map_height = env.grid_size * cell_size
        
        self.panel_height = 230 
        
        self.width = max(self.map_width, 500) 
        self.height = self.map_height + self.panel_height
        
        # محاسبه فاصله افقی برای قرار دادن نقشه دقیقاً در وسط صفحه
        self.offset_x = (self.width - self.map_width) // 2
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Dynamic Maze RL - Centered Visualizer")
        
        self._init_fonts()

    def _init_fonts(self):
        try:
            self.font_main = pygame.font.SysFont("Arial", 14)
            self.font_bold = pygame.font.SysFont("Arial", 14, bold=True)
            self.font_title = pygame.font.SysFont("Arial", 15, bold=True)
            self.font_key = pygame.font.SysFont("Arial", 16, bold=True)
        except:
            self.font_main = pygame.font.Font(None, 22)
            self.font_bold = pygame.font.Font(None, 22)
            self.font_title = pygame.font.Font(None, 24)
            self.font_key = pygame.font.Font(None, 24)

    def _get_cell_color(self, cell_type):
        if cell_type == self.env.WALL: return COLORS["wall"]
        if cell_type == self.env.PENALTY: return COLORS["penalty"]
        if cell_type == self.env.GOAL: return COLORS["goal"]
        if cell_type == self.env.KEY and not self.env.has_key: return COLORS["key"]
        if cell_type == self.env.DOOR: return COLORS["door_open"] if self.env.has_key else COLORS["door_locked"]
        return COLORS["path"]

    def _draw_grid(self):
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                # اضافه شدن offset_x برای وسط‌چین کردن خانه‌ها
                rect = pygame.Rect(self.offset_x + c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                cell_type = self.env.grid[r, c]
                
                pygame.draw.rect(self.screen, self._get_cell_color(cell_type), rect)
                pygame.draw.rect(self.screen, COLORS["grid_line"], rect, 1)

                if cell_type == self.env.KEY and not self.env.has_key:
                    key_surf = self.font_key.render("K", True, COLORS["text_dark"])
                    key_rect = key_surf.get_rect(center=rect.center)
                    self.screen.blit(key_surf, key_rect)

    def _draw_policy_arrows(self, policy_map):
        if not policy_map: return
        offset = self.cell_size // 3
        for (r, c, has_key), action in policy_map.items():
            if self.env.grid[r, c] == self.env.WALL: continue
            
            # وسط‌چین شدن فلش‌ها
            center = (self.offset_x + c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)
            end = center
            if action == 0:   end = (center[0], center[1] - offset)
            elif action == 1: end = (center[0] + offset, center[1])
            elif action == 2: end = (center[0], center[1] + offset)
            elif action == 3: end = (center[0] - offset, center[1])
            
            pygame.draw.line(self.screen, COLORS["arrow"], center, end, 2)
            pygame.draw.circle(self.screen, COLORS["arrow"], end, 2)

    def _draw_dynamic_obstacle(self, state):
        if len(state) >= 4:
            patrol_idx = state[3]
            pr, pc = self.env.patrol_route[patrol_idx]
            # وسط‌چین شدن مانع
            rect = pygame.Rect(self.offset_x + pc * self.cell_size + 4, pr * self.cell_size + 4, self.cell_size - 8, self.cell_size - 8)
            pygame.draw.rect(self.screen, COLORS["obstacle"], rect)

    def _draw_agent(self, state):
        ar, ac = state[0], state[1]
        # وسط‌چین شدن عامل
        center = (self.offset_x + ac * self.cell_size + self.cell_size // 2, ar * self.cell_size + self.cell_size // 2)
        pygame.draw.circle(self.screen, COLORS["agent"], center, self.cell_size // 3)

    def _draw_dashboard(self, info):
        panel_y = self.map_height
        
        pygame.draw.rect(self.screen, COLORS["panel_bg"], (0, panel_y, self.width, self.panel_height))
        pygame.draw.line(self.screen, COLORS["panel_border"], (0, panel_y), (self.width, panel_y), 2)

        x_offset = 20
        y_offset = panel_y + 15
        
        info = info or {}
        status_val = info.get('status', 'Running')
        key_status = "Acquired (1)" if info.get('key', 0) == 1 else "Not Acquired (0)"
        
        self.screen.blit(self.font_title.render("--- Simulation Status ---", True, COLORS["text_blue"]), (x_offset, y_offset))
        y_offset += 25
        
        lines_info = [
            f"Algorithm: {info.get('algorithm', 'Standard Environment')}",
            f"Episode: {info.get('episode', 0)}    |    Step: {info.get('step', 0)}",
            f"Total Reward: {info.get('reward', 0)}    |    Key: {key_status}"
        ]
        
        for text in lines_info:
            self.screen.blit(self.font_bold.render(text, True, COLORS["text_dark"]), (x_offset, y_offset))
            y_offset += 22

        status_color = COLORS["text_green"] if "Goal" in status_val else COLORS["text_red"] if "Failed" in status_val else COLORS["text_dark"]
        self.screen.blit(self.font_bold.render(f"Status: {status_val}", True, status_color), (x_offset, y_offset))
        y_offset += 26
        
        self.screen.blit(self.font_title.render("--- Keyboard Controls ---", True, COLORS["text_blue"]), (x_offset, y_offset))
        y_offset += 25
        
        col1 = [
            "[SPACE]: Pause / Resume",
            "[R]: Reset Episode",
            "[P]: Toggle Policy View"
        ]
        col2 = [
            "[A]: Switch Algorithm",
            "[UP / DOWN]: Adjust Speed"
        ]
        
        temp_y = y_offset
        for ctrl in col1:
            self.screen.blit(self.font_main.render(ctrl, True, COLORS["text_gray"]), (x_offset, temp_y))
            temp_y += 22
            
        temp_y = y_offset
        col2_x = x_offset + 220 
        for ctrl in col2:
            self.screen.blit(self.font_main.render(ctrl, True, COLORS["text_gray"]), (col2_x, temp_y))
            temp_y += 22

    def draw_state(self, state, info=None, policy_map=None, *args, **kwargs):
        self.screen.fill(COLORS["background"])
        
        self._draw_grid()
        self._draw_policy_arrows(policy_map)
        self._draw_dynamic_obstacle(state)
        self._draw_agent(state)
        self._draw_dashboard(info)
        
        pygame.display.flip()