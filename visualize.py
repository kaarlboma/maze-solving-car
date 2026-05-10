import pygame
import numpy as np
import random
from GridWorld import GridWorld, Agent, train as rl_train
from agent_dqn import DQN, ReplayBuffer, train_dqn

# ── Palette ────────────────────────────────────────────────────────────────────
BG        = (28,  28,  38)
PANEL     = (40,  40,  55)
FLOOR     = (235, 235, 242)
FLOOR_HOV = (210, 220, 255)   # hover tint on floor cells
START_CLR = (255, 225,  80)   # start cell highlight
WALL_CLR  = (55,  55,  70)
GOAL_CLR  = (72,  199, 116)
AGENT_CLR = (66,  135, 245)
ARROW_CLR = (50,  80,  210)
TEXT_LT   = (210, 210, 225)
TEXT_DK   = (30,  30,  45)
BTN_BASE  = (58,  58,  78)
BTN_SEL   = (66,  135, 245)
BTN_OK    = (72,  199, 116)
BTN_STOP  = (220,  70,  70)
BTN_RAND  = (180,  90, 220)
ACCENT    = (66,  135, 245)
MUTED     = (140, 140, 170)

# ── Layout ─────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 820, 600
SB_W         = 210
TOP_H        = 56
PAD          = 18
MAX_CELL     = 120
STEP_MS      = 480
FPS          = 60

# ── Preset worlds ──────────────────────────────────────────────────────────────
worlds = {
    "2×2": (np.array([['-', 'X'],
                      ['-', 'G']]), (0, 0)),
    "3×3": (np.array([['-', '-', '-'],
                      ['-', 'X', '-'],
                      ['-', '-', 'G']]), (0, 0)),
    "4×4": (np.array([['-', '-', '-', '-'],
                      ['-', 'X', 'X', '-'],
                      ['-', '-', '-', '-'],
                      ['-', 'X', '-', 'G']]), (0, 0)),
    "5×5": (np.array([['-', '-', '-', '-', '-'],
                      ['-', 'X', 'X', '-', '-'],
                      ['-', '-', '-', 'X', '-'],
                      ['-', 'X', '-', '-', '-'],
                      ['-', '-', '-', 'X', 'G']]), (0, 0)),
}
WORLD_KEYS = list(worlds.keys())
ACTIONS    = ['U', 'D', 'L', 'R']
REWARDS    = {'-': -1, 'G': 10}


# ── Standalone helpers ─────────────────────────────────────────────────────────

def draw_arrow(surf, color, cx, cy, direction, half):
    h = half
    if direction == 'U':
        pts = [(cx, cy - h), (cx - h*2//3, cy + h//2), (cx + h*2//3, cy + h//2)]
    elif direction == 'D':
        pts = [(cx, cy + h), (cx - h*2//3, cy - h//2), (cx + h*2//3, cy - h//2)]
    elif direction == 'L':
        pts = [(cx - h, cy), (cx + h//2, cy - h*2//3), (cx + h//2, cy + h*2//3)]
    else:
        pts = [(cx + h, cy), (cx - h//2, cy - h*2//3), (cx - h//2, cy + h*2//3)]
    pygame.draw.polygon(surf, color, pts)


def best_action(agent, state, mode):
    if mode == 'dqn':
        return agent.choose_action(epsilon=0, state=state)
    return max(ACTIONS, key=lambda a: agent.qtable[(state, a)])


def reachable(grid, start, goal):
    rows, cols = grid.shape
    visited, queue = set(), [start]
    while queue:
        cur = queue.pop(0)
        if cur == goal:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        r, c = cur
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr][nc] != 'X'):
                queue.append((nr, nc))
    return False


def random_world(rows, cols, start):
    goal = (rows - 1, cols - 1)
    for _ in range(300):
        grid = np.full((rows, cols), '-')
        grid[goal[0]][goal[1]] = 'G'
        for r in range(rows):
            for c in range(cols):
                if (r, c) != start and (r, c) != goal:
                    if random.random() < 0.30:
                        grid[r][c] = 'X'
        if reachable(grid, start, goal):
            return grid
    # fallback: no walls
    grid = np.full((rows, cols), '-')
    grid[goal[0]][goal[1]] = 'G'
    return grid


# ── App ────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        self.sw, self.sh = WIN_W, WIN_H
        pygame.display.set_caption("GridWorld – Q-Learning")
        self.clock = pygame.time.Clock()

        self.f_lg = pygame.font.Font(None, 28)
        self.f_md = pygame.font.Font(None, 22)
        self.f_sm = pygame.font.Font(None, 17)

        self.world_idx  = 2
        self.trained    = False
        self.agent      = None
        self.animating  = False
        self.anim_path  = []
        self.anim_step  = 0
        self.anim_t     = 0
        self.status     = "Click a floor cell to set start · then Train"

        self.mode        = 'ql'
        self.world_btns  = []
        self.train_btn   = pygame.Rect(0, 0, 0, 0)
        self.run_btn     = pygame.Rect(0, 0, 0, 0)
        self.rand_btn    = pygame.Rect(0, 0, 0, 0)
        self.mode_btn    = pygame.Rect(0, 0, 0, 0)

        self._load_world()

    # ── World setup ────────────────────────────────────────────────────────────

    def _load_world(self):
        name = WORLD_KEYS[self.world_idx]
        arr, start = worlds[name]
        self.world_arr = arr.copy()
        self.start = start
        self.rows, self.cols = arr.shape
        self._recalc_layout()
        self._reset_training()

    def _recalc_layout(self):
        area_w = self.sw - SB_W - 2 * PAD
        area_h = self.sh - TOP_H - 2 * PAD
        self.cell = min(MAX_CELL, area_w // self.cols, area_h // self.rows)
        total_w = self.cols * self.cell
        total_h = self.rows * self.cell
        self.gx = PAD + (area_w - total_w) // 2
        self.gy = TOP_H + PAD + (area_h - total_h) // 2

    def _reset_training(self):
        self.trained   = False
        self.agent     = None
        self.animating = False
        self.anim_path = []
        self.status    = "Click a floor cell to set start · then Train"

    def _cell_rect(self, r, c):
        return pygame.Rect(self.gx + c * self.cell,
                           self.gy + r * self.cell,
                           self.cell, self.cell)

    def _grid_cell_at(self, mx, my):
        """Return (r, c) of the grid cell under pixel (mx, my), or None."""
        for r in range(self.rows):
            for c in range(self.cols):
                if self._cell_rect(r, c).collidepoint(mx, my):
                    return r, c
        return None

    # ── Randomize ─────────────────────────────────────────────────────────────

    def _randomize(self):
        self.world_arr = random_world(self.rows, self.cols, self.start)
        self._reset_training()
        self.status = "Randomized!  Click Train"

    # ── Training ───────────────────────────────────────────────────────────────

    def _train(self):
        self.status = "Training…"
        self._draw()
        pygame.display.flip()
        gw = GridWorld(self.world_arr.copy(), self.start, REWARDS, 0.9)
        if self.mode == 'dqn':
            ag = DQN(ACTIONS)
            buf = ReplayBuffer(capacity=10000)
            train_dqn(gw, ag, buf, episodes=5000, epsilon=0.3, alpha=0.001, gamma=0.9)
        else:
            ag = Agent(gw, ACTIONS)
            rl_train(gw, ag, episodes=5000, epsilon=0.3, alpha=0.1, gamma=0.9)
        self.agent   = ag
        self.trained = True
        self.status  = "Trained!  Click Run to watch the agent"

    # ── Animation ─────────────────────────────────────────────────────────────

    def _build_path(self):
        gw   = GridWorld(self.world_arr.copy(), self.start, REWARDS, 0.9)
        path = [gw.current_pos]
        for _ in range(200):
            a = best_action(self.agent, gw.current_pos, self.mode)
            pos, _, done = gw.step(a)
            path.append(pos)
            if done:
                break
        return path

    def _start_anim(self):
        self.anim_path = self._build_path()
        self.anim_step = 0
        self.anim_t    = pygame.time.get_ticks()
        self.animating = True
        self.status    = "Running…"

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _draw_topbar(self):
        pygame.draw.rect(self.screen, PANEL, (0, 0, self.sw, TOP_H))
        pygame.draw.line(self.screen, BG, (0, TOP_H), (self.sw, TOP_H), 2)
        title = self.f_lg.render("GridWorld  ·  Q-Learning", True, TEXT_LT)
        self.screen.blit(title, (PAD, TOP_H // 2 - title.get_height() // 2))
        status = self.f_md.render(self.status, True, ACCENT)
        self.screen.blit(status, status.get_rect(
            midright=(self.sw - SB_W - PAD, TOP_H // 2)))

    def _draw_grid(self):
        mx, my = pygame.mouse.get_pos()
        hover  = self._grid_cell_at(mx, my)

        for r in range(self.rows):
            for c in range(self.cols):
                rect = self._cell_rect(r, c)
                cell = self.world_arr[r][c]

                if cell == 'X':
                    bg = WALL_CLR
                elif cell == 'G':
                    bg = GOAL_CLR
                elif (r, c) == self.start:
                    bg = START_CLR
                elif hover == (r, c) and not self.animating:
                    bg = FLOOR_HOV
                else:
                    bg = FLOOR

                pygame.draw.rect(self.screen, bg, rect)
                pygame.draw.rect(self.screen, BG, rect, 2)

                cx, cy = rect.centerx, rect.centery
                half   = self.cell // 5

                if cell == 'G':
                    lbl = self.f_lg.render("G", True, (255, 255, 255))
                    self.screen.blit(lbl, lbl.get_rect(center=(cx, cy)))
                elif cell == '-':
                    if (r, c) == self.start:
                        lbl = self.f_sm.render("S", True, TEXT_DK)
                        self.screen.blit(lbl, (rect.x + 5, rect.y + 4))
                    if self.trained:
                        draw_arrow(self.screen, ARROW_CLR, cx, cy,
                                   best_action(self.agent, (r, c), self.mode), half)

        # agent circle
        if self.animating and self.anim_step < len(self.anim_path):
            r, c = self.anim_path[self.anim_step]
            rect = self._cell_rect(r, c)
            pygame.draw.circle(self.screen, AGENT_CLR, rect.center, self.cell // 3)
            pygame.draw.circle(self.screen, (255, 255, 255), rect.center,
                               self.cell // 3, 3)

    def _draw_sidebar(self):
        sbx = self.sw - SB_W
        pygame.draw.rect(self.screen, PANEL, (sbx, 0, SB_W, self.sh))
        pygame.draw.line(self.screen, BG, (sbx, 0), (sbx, self.sh), 2)

        y = TOP_H + 14
        self.screen.blit(self.f_lg.render("World", True, TEXT_LT), (sbx + 16, y))
        y += 26

        self.world_btns = []
        for i, name in enumerate(WORLD_KEYS):
            btn = pygame.Rect(sbx + 12, y, SB_W - 24, 34)
            col = BTN_SEL if i == self.world_idx else BTN_BASE
            pygame.draw.rect(self.screen, col, btn, border_radius=7)
            lbl = self.f_md.render(name, True, TEXT_LT)
            self.screen.blit(lbl, lbl.get_rect(center=btn.center))
            self.world_btns.append((btn, i))
            y += 41

        y += 8

        self.mode_btn = pygame.Rect(sbx + 12, y, SB_W - 24, 34)
        mode_label = "Mode: Q-Learn" if self.mode == 'ql' else "Mode: DQN"
        pygame.draw.rect(self.screen, BTN_RAND, self.mode_btn, border_radius=8)
        lbl = self.f_md.render(mode_label, True, (255, 255, 255))
        self.screen.blit(lbl, lbl.get_rect(center=self.mode_btn.center))
        y += 42

        self.rand_btn = pygame.Rect(sbx + 12, y, SB_W - 24, 34)
        pygame.draw.rect(self.screen, BTN_RAND, self.rand_btn, border_radius=8)
        lbl = self.f_lg.render("Randomize", True, (255, 255, 255))
        self.screen.blit(lbl, lbl.get_rect(center=self.rand_btn.center))
        y += 42

        self.train_btn = pygame.Rect(sbx + 12, y, SB_W - 24, 34)
        pygame.draw.rect(self.screen, ACCENT, self.train_btn, border_radius=8)
        lbl = self.f_lg.render("Train", True, (255, 255, 255))
        self.screen.blit(lbl, lbl.get_rect(center=self.train_btn.center))
        y += 42

        self.run_btn = pygame.Rect(sbx + 12, y, SB_W - 24, 34)
        if self.animating:
            run_col, run_label = BTN_STOP, "Stop"
        elif self.trained:
            run_col, run_label = BTN_OK, "Run"
        else:
            run_col, run_label = BTN_BASE, "Run"
        pygame.draw.rect(self.screen, run_col, self.run_btn, border_radius=8)
        lbl = self.f_lg.render(run_label, True, (255, 255, 255))
        self.screen.blit(lbl, lbl.get_rect(center=self.run_btn.center))
        y += 50

        self.screen.blit(self.f_md.render("Hyperparameters", True, TEXT_LT),
                         (sbx + 16, y))
        y += 22
        for line in ["α  =  0.1   (learning rate)",
                     "γ  =  0.9   (discount)",
                     "ε  =  0.3   (exploration)",
                     "5 000 episodes"]:
            self.screen.blit(self.f_sm.render(line, True, MUTED), (sbx + 16, y))
            y += 18

        # hint
        y += 10
        hint = self.f_sm.render("Click grid to move start (S)", True, MUTED)
        self.screen.blit(hint, (sbx + 16, y))

    def _draw(self):
        self.screen.fill(BG)
        self._draw_topbar()
        self._draw_grid()
        self._draw_sidebar()

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _step(self):
        """Process one frame. Returns False when the app should quit."""
        sw, sh = self.screen.get_size()
        if (sw, sh) != (self.sw, self.sh):
            self.sw, self.sh = sw, sh
            self._recalc_layout()

        now = pygame.time.get_ticks()

        if self.animating:
            if self.anim_step < len(self.anim_path) - 1:
                if now - self.anim_t >= STEP_MS:
                    self.anim_step += 1
                    self.anim_t = now
            else:
                self.animating = False
                self.status = "Reached the goal!  Click Run to replay"

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return False

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
                if ev.key == pygame.K_t:
                    self._train()
                if ev.key == pygame.K_r and self.trained:
                    self._start_anim()

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos

                for btn, idx in self.world_btns:
                    if btn.collidepoint(mx, my):
                        self.world_idx = idx
                        self._load_world()
                if self.mode_btn.collidepoint(mx, my):
                    self.mode = 'dqn' if self.mode == 'ql' else 'ql'
                    self._reset_training()
                    self.status = f"Mode: {'DQN' if self.mode == 'dqn' else 'Q-Learning'}  · Click Train"
                if self.rand_btn.collidepoint(mx, my):
                    self._randomize()
                if self.train_btn.collidepoint(mx, my):
                    self._train()
                if self.run_btn.collidepoint(mx, my):
                    if self.animating:
                        self.animating = False
                        self.status = "Stopped.  Click Run to replay"
                    elif self.trained:
                        self._start_anim()

                cell = self._grid_cell_at(mx, my)
                if cell is not None:
                    r, c = cell
                    if self.world_arr[r][c] == '-':
                        self.start = (r, c)
                        self._reset_training()
                        self.status = "Start moved to " + str(cell) + "  · Click Train"

        self._draw()
        pygame.display.flip()
        self.clock.tick(FPS)
        return True

    def run(self):
        while self._step():
            pass


if __name__ == "__main__":
    App().run()
