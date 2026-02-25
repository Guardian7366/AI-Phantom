from cmath import phase

import pygame
import math

from ai_phantom.envs.maze.maze_env import MazeConfig, MazeEnv
from ai_phantom.planners.bfs import bfs_plan, path_to_actions
from utils.start_menu import Button, Icon_Button, SettingsPanel
from utils.conf import WINDOW_WIDTH, FPS, Config


class MazeTrainingScreen:
    """
    Pantalla de entrenamiento del laberinto (UI).
    - Compatible con las fuentes y tamaños de StartScreen.
    - Usa SettingsState para sincronizar volúmenes y fullscreen.
    """

    def __init__(self, config: Config):
        self.clock = config.clock
        self.settings = config.settings
        self.screen = config.screen
        self.click_sound = config.click_sound
        self.font_title = config.font_title
        self.font_statsTitle = config.font_statsTitle
        self.font_button = config.font_button
        self.font_text = config.font_text

        self.running = True

        #Playback state
        self.playing = False
        self.speed_index = 0
        self.speeds = [1, 2, 4]  # x1, x2, x4
        self.sleeps = [500, 250, 125]  # ms per step for each speed
        self.current_tick = 0

        #Flags
        self.show_settings = False

        self.maze_cfg = MazeConfig(
            height=12,
            width=12,
            use_walls=True,
            max_steps=256,
            min_manhattan=6,
        )
        self.maze_env = MazeEnv(self.maze_cfg, seed=0)
        self.maze_grid = None
        self.maze_actions = None
        self.current_action = "idle"

        #Create buttons
        self._create_buttons()
        #Compute layout
        self._recalc_layout()

        #Apply shared sound settings
        self.settings.apply_music_volume()
        self.settings.apply_sfx_volume(self.click_sound)

        #Storage for arrow rects used in settings overlay
        self._settings_arrow_rects = {}
        self.settings_panel = SettingsPanel(self.screen, self.settings, self.click_sound, self.font_button)

        # Set images according to ACTIONS in maze env
        self.ghost_img = {}
        self.ghost_img[0] = pygame.image.load("assets/sprites/phantom/PhantomBack.png").convert_alpha()
        self.ghost_img[1] = pygame.image.load("assets/sprites/phantom/PhantomFront.png").convert_alpha()
        self.ghost_img[2] = pygame.image.load("assets/sprites/phantom/PhantomLeft.png").convert_alpha()
        self.ghost_img[3] = pygame.image.load("assets/sprites/phantom/PhantomRight.png").convert_alpha()
        self.ghost_img["idle"] = pygame.image.load("assets/sprites/phantom/PhantomIdle.png").convert_alpha()
        self.floor_img = pygame.image.load("assets/sprites/misc/FloorMaze.png").convert()
        self.wall_img = pygame.image.load("assets/sprites/misc/Wall.png").convert()
        self.goal_img = pygame.image.load("assets/sprites/misc/GoalPanel.png").convert()

        self.episode_num = 0

    # ----------------------
    # CREATION & LAYOUT
    # ----------------------
    def _create_buttons(self):
        #Main buttons
        self.btn_back = Button((20, 20, 140, 50), "BACK", self.font_button, (60, 60, 90), (90, 90, 140), click_sound=self.click_sound)
        self.btn_settings = Icon_Button((WINDOW_WIDTH - 160, 20, 75, 75), "assets/images/gear.png", self.font_button, (40, 40, 60), (80, 80, 120), click_sound=self.click_sound)
        #Play bbutton
        self.btn_play = Button((0, 0, 200, 60), "PLAY", self.font_button, (40, 120, 40), (60, 160, 60), click_sound=self.click_sound)
        self.btn_ff = Button((0, 0, 120, 44), "x1", self.font_button, (60, 60, 90), (100, 100, 140), click_sound=self.click_sound)

    def _recalc_layout(self):
        width, height = self.screen.get_size()

        margin = 40
        gap = 24

        total_w = width - 2 * margin
        maze_w = int(total_w * 0.68)
        stats_w = total_w - maze_w - gap

        area_top = 110
        area_h = height - area_top - 70

        #Rects
        self.maze_rect = pygame.Rect(margin, area_top, maze_w, area_h)
        self.stats_rect = pygame.Rect(margin + maze_w + gap, area_top, stats_w, area_h)

        #Title center
        self.title_pos = (width // 2, 48)

        #Control positions
        ctl_y = area_top + 10
        play_w, play_h = 170, 44
        ff_w, ff_h = 120, 44

        play_x = ff_x = self.stats_rect.left + 12

        self.btn_play.rect.topleft = (play_x, ctl_y)
        self.btn_play.rect.size = (play_w, play_h)

        self.btn_ff.rect.topleft = (ff_x, ctl_y + play_h + 12)
        self.btn_ff.rect.size = (ff_w, ff_h)

        self.btn_back.rect.topleft = (20, 30)
        self.btn_back.rect.size = (140, 50)

        self.btn_settings.rect.topright = (width - 20, 20)
        self.btn_settings.rect.size = (75, 75)

    # ----------------------------------
    # DRAW SCREEN ELEMENTS
    # ----------------------------------
    def _draw_title(self):
        title = self.font_title.render("TRAINING", False, (255, 255, 255))
        rect = title.get_rect(center=self.title_pos)
        self.screen.blit(title, rect)

    def _draw_maze_area(self):
        if self.maze_grid is None:
            return

        pygame.draw.rect(self.screen, (12, 14, 22), self.maze_rect)
        pygame.draw.rect(self.screen, (90, 90, 90), self.maze_rect, 3)
        pygame.draw.rect(self.screen, (150, 150, 150), self.maze_rect, 2)

        cols = self.maze_cfg.width
        rows = self.maze_cfg.height
        cell_w = self.maze_rect.width / cols
        cell_h = self.maze_rect.height / rows

        for r in range(rows):
            for c in range(cols):
                x = int(self.maze_rect.left + c * cell_w)
                y = int(self.maze_rect.top + r * cell_h)
                rect = pygame.Rect(x, y, math.ceil(cell_w), math.ceil(cell_h))
                image = None
                char = self.maze_grid[r][c]
                if char == "#":
                    image = self.wall_img
                elif char == "G":
                    image = self.goal_img
                elif char == "A":
                    image = self.ghost_img.get(self.current_action, self.ghost_img["idle"])
                else:
                    image = self.floor_img
                
                # Dibujar piso primero
                floor_scaled = pygame.transform.scale(self.floor_img, (rect.width, rect.height))
                self.screen.blit(floor_scaled, rect)
                # Luego dibujar el sprite correspondiente encima
                image = pygame.transform.scale(image, (rect.width, rect.height))
                self.screen.blit(image, rect)

    def _draw_stats_area(self):
        pygame.draw.rect(self.screen, (16, 18, 26), self.stats_rect)
        pygame.draw.rect(self.screen, (120, 120, 120), self.stats_rect, 2)

        hdr = self.font_text.render("Statistics", False, (230, 230, 230))
        self.screen.blit(hdr, (self.stats_rect.left + 12, self.btn_ff.rect.bottom + 24))

        y = self.btn_ff.rect.bottom + hdr.get_height() + 36
        lines = [
            f"Playing: {'Yes' if self.playing else 'No'}",
            f"Speed: x{self.speeds[self.speed_index]}",
            f"Episode: {self.episode_num}",
            "Mean Reward: -",
            "Best Success: -",
            "",
        ]
        for ln in lines:
            surf = self.font_text.render(ln, False, (200, 200, 200))
            self.screen.blit(surf, (self.stats_rect.left + 12, y))
            y += 30

    def _draw_controls(self):
        mouse_pos = pygame.mouse.get_pos()
        #Update/draw top controls
        for b in (self.btn_back, self.btn_settings, self.btn_play, self.btn_ff):
            b.update(mouse_pos)
            b.draw(self.screen)

        #Synchronize labels
        self.btn_play.text = "PAUSE" if self.playing else "PLAY"
        self.btn_ff.text = f"x{self.speeds[self.speed_index]}"

    # ----------------------------------
    # ELEMENT INTERACTION
    # ----------------------------------
    def _handle_control_click(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN:
            return None
        pos = event.pos

        if self.btn_back.is_clicked(event):
            #Return value to move to selection menu in main
            return "selection"

        if self.btn_settings.is_clicked(event):
            #Display setting panel
            self.show_settings = True
            return None

        if self.btn_play.is_clicked(event):
            #Switch Play button mode
            self.playing = not self.playing
            return None

        if self.btn_ff.is_clicked(event):
            #Switch through simulation speeds
            self.speed_index = (self.speed_index + 1) % len(self.speeds)
            return None

        return None

    # ----------------------------------
    # MAIN LOOP
    # ----------------------------------
    def run(self):
        while self.running:
            self.clock.tick(FPS)
            # Recompute layout only if size changed
            self._recalc_layout()

            if self.maze_grid is None:
                obs, info = self.maze_env.reset(seed=0, phase=0)
                self.maze_grid = self.maze_env.render().splitlines()

            if self.maze_actions is None:
                self.current_tick = pygame.time.get_ticks()
                path = bfs_plan(self.maze_env.walls, self.maze_env.agent, self.maze_env.goal)
                if path is not None:
                    self.maze_actions = path_to_actions(path)
                else:
                    self.maze_grid = None  # Trigger new maze generation if no path found
            else:
                # Step through actions at the current speed when playing
                if self.playing and pygame.time.get_ticks() - self.current_tick > self.sleeps[self.speed_index]:
                    self.current_tick = pygame.time.get_ticks()
                    # Step the maze_env with the next action
                    self.current_action = self.maze_actions.pop(0)
                    # Update the maze environment and grid 
                    obs, reward, done, info = self.maze_env.step(self.current_action)
                    self.maze_grid = self.maze_env.render().splitlines()
                    if done:
                        self.current_action = "idle"  # Reset action to idle when episode ends
                        self.maze_actions = None  # Reset for next episode
                        self.maze_grid = None  # Trigger new maze generation
                        self.episode_num += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                if self.show_settings:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if self.settings_panel.handle_settings_click(event):
                            self.show_settings = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.show_settings = False
                else:
                    #Interaction
                    res = self._handle_control_click(event)
                    if res == "selection":
                        return "selection"

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return "selection"

                    #Clicks inside canvas or stats (placeholder feedback)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if self.maze_rect.collidepoint(event.pos) or self.stats_rect.collidepoint(event.pos):
                            if self.click_sound:
                                self.click_sound.play()

            #Draw screen elements
            self.screen.fill((10, 12, 18))
            self._draw_title()
            self._draw_maze_area()
            self._draw_stats_area()
            self._draw_controls()

            if self.show_settings:
                overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                self.screen.blit(overlay, (0, 0))
                self.settings_panel.draw_settings_panel()

            pygame.display.flip()

        return None
