from cmath import phase

import pygame
import math

from ai_phantom.envs.maze.maze_env import MazeConfig, MazeEnv
from ai_phantom.planners.bfs import bfs_plan, path_to_actions
from utils.start_menu import Button, Icon_Button, SettingsPanel
from utils.conf import WINDOW_WIDTH, FPS, Config


class MazeGameScreen:
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
        self.player_img = pygame.image.load("assets/sprites/misc/Player.png").convert_alpha()

    # ----------------------
    # CREATION & LAYOUT
    # ----------------------
    def _create_buttons(self):
        #Main buttons
        self.btn_back = Button((20, 20, 140, 50), "BACK", self.font_button, (60, 60, 90), (90, 90, 140), click_sound=self.click_sound)
        self.btn_settings = Icon_Button((WINDOW_WIDTH - 160, 20, 75, 75), "assets/images/gear.png", self.font_button, (40, 40, 60), (80, 80, 120), click_sound=self.click_sound)
        #Play bbutton
        #self.btn_play = Button((0, 0, 200, 60), "PLAY", self.font_button, (40, 120, 40), (60, 160, 60), click_sound=self.click_sound)

    def _recalc_layout(self):
        width, height = self.screen.get_size()

        margin = width / 6
        gap = 24

        total_w = width - 2
        maze_w = int(total_w * 0.68)
        stats_w = total_w - maze_w - gap

        area_top = 110
        area_h = height - area_top - 70

        #Rects
        self.maze_rect = pygame.Rect(margin, area_top, maze_w, area_h)
        self.stats_rect = pygame.Rect(margin + maze_w + gap, area_top, stats_w, area_h)

        #Title center
        self.title_pos = (width // 2, 48)

        #Ghost and score title
        self.ghost_title_pos = (width // 4, 30)
        self.ghost_score_pos = (width // 4, 65)

        #Player and score title
        self.player_title_pos = (width * 0.75, 30)
        self.player_score_pos = (width * 0.75, 65)

        #Control positions
        ctl_y = area_top + 10
        play_w, play_h = 170, 44
        ff_w, ff_h = 120, 44

        play_x = ff_x = self.stats_rect.left + 12

        #self.btn_play.rect.topleft = (play_x, ctl_y)
        #self.btn_play.rect.size = (play_w, play_h)

        self.btn_back.rect.topleft = (20, 30)
        self.btn_back.rect.size = (140, 50)

        self.btn_settings.rect.topright = (width - 20, 20)
        self.btn_settings.rect.size = (75, 75)

        #btn_back_overlay will be positioned inside the overlay panel when drawing

    # ----------------------------------
    # DRAW SCREEN ELEMENTS
    # ----------------------------------
    def _draw_title(self):
        title = self.font_title.render("MAZE", False, (255, 255, 255))
        rect = title.get_rect(center=self.title_pos)
        self.screen.blit(title, rect)

    def _draw_maze_area(self):
        if self.maze_grid is None:
            return

        pygame.draw.rect(self.screen, (12, 14, 22), self.maze_rect)
        pygame.draw.rect(self.screen, (90, 90, 90), self.maze_rect, 3)
        pygame.draw.rect(self.screen, (150, 150, 150), self.maze_rect, 2)

        # Dimensiones del grid
        cols = self.maze_cfg.width
        rows = self.maze_cfg.height
        cell_w = self.maze_rect.width / cols
        cell_h = self.maze_rect.height / rows

        ghost_row = None
        ghost_col = None

        # ===== 1. DIBUJAR MAPA (SIN FANTASMA) =====
        for r in range(rows):
            for c in range(cols):
                x = int(self.maze_rect.left + c * cell_w)
                y = int(self.maze_rect.top + r * cell_h)
                rect = pygame.Rect(x, y, math.ceil(cell_w), math.ceil(cell_h))

                char = self.maze_grid[r][c]

                if char == "A":
                    # Guardamos posición del fantasma pero NO lo dibujamos aquí
                    ghost_row = r
                    ghost_col = c
                    image = self.floor_img  # piso debajo del fantasma
                elif char == "#":
                    image = self.wall_img
                elif char == "G":
                    # Dibujar piso primero
                    floor_scaled = pygame.transform.scale(self.floor_img, (rect.width, rect.height))
                    self.screen.blit(floor_scaled, rect)

                    # Luego dibujar el goal encima (con transparencia)
                    goal_scaled = pygame.transform.scale(self.goal_img, (rect.width, rect.height))
                    self.screen.blit(goal_scaled, rect)
                    continue
                else:
                    image = self.floor_img

                scaled = pygame.transform.scale(image, (rect.width, rect.height))
                self.screen.blit(scaled, rect)

        # ===== 2. DIBUJAR FANTASMA ENCIMA (CAPA SUPERIOR) =====
        if ghost_row is not None and ghost_col is not None:
            ghost_img = self.ghost_img.get(
                self.current_action,
                self.ghost_img["idle"]
            )

            ghost_w = math.ceil(cell_w)
            ghost_h = math.ceil(cell_h)

            # smoothscale = mejor calidad y menos halo
            ghost_scaled = pygame.transform.smoothscale(ghost_img, (ghost_w, ghost_h))

            ghost_x = int(self.maze_rect.left + ghost_col * cell_w)
            ghost_y = int(self.maze_rect.top + ghost_row * cell_h)

            self.screen.blit(ghost_scaled, (ghost_x, ghost_y))

    def _draw_controls(self):
        mouse_pos = pygame.mouse.get_pos()
        #Update/draw top controls
        for b in (self.btn_back, self.btn_settings):
            b.update(mouse_pos)
            b.draw(self.screen)


        ghost_title = self.font_button.render("GHOST", False, (230, 230, 230))
        rectGT = ghost_title.get_rect(center=self.ghost_title_pos)
        self.screen.blit(ghost_title, rectGT)

        ghost_score = self.font_button.render("0", False, (230, 230, 230))
        rectGS = ghost_score.get_rect(center=self.ghost_score_pos)
        self.screen.blit(ghost_score, rectGS)

        player_title = self.font_button.render("PLAYER", False, (230, 230, 230))
        rectPT = player_title.get_rect(center=self.player_title_pos)
        self.screen.blit(player_title, rectPT)

        player_score = self.font_button.render("0", False, (230, 230, 230))
        rectPS = player_score.get_rect(center=self.player_score_pos)
        self.screen.blit(player_score, rectPS)


        #Synchronize labels
        #self.btn_play.text = "PAUSE" if self.playing else "PLAY"

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
        """
        if self.btn_play.is_clicked(event):
            #Switch Play button mode
            self.playing = not self.playing
            return None
        """
        return None
    
    def move_ghost(self):
        if self.maze_actions is None:
            self.current_tick = pygame.time.get_ticks()
            path = bfs_plan(self.maze_env.walls, self.maze_env.agent, self.maze_env.goal)
            if path is not None:
                self.maze_actions = path_to_actions(path)
            else:
                self.maze_grid = None  # Trigger new maze generation if no path found
        else:
            # Step through actions at the current speed when playing
            if pygame.time.get_ticks() - self.current_tick > self.sleeps[self.speed_index]:
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
                self.maze_env.goal = None  # Clear goal to allow player placement
                self.maze_grid = self.maze_env.render().splitlines()

            if self.maze_env.goal is not None:
                self.move_ghost()  # Start the ghost movement logic

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                if self.maze_env.goal is None:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        # User puts player with click
                        mouse_pos = pygame.mouse.get_pos()
                        # Check that mouse is inside maze area
                        if self.maze_rect.collidepoint(mouse_pos):
                            # Get grid cell from mouse position
                            rel_x = mouse_pos[0] - self.maze_rect.left
                            rel_y = mouse_pos[1] - self.maze_rect.top
                            cell_w = self.maze_rect.width / self.maze_cfg.width
                            cell_h = self.maze_rect.height / self.maze_cfg.height
                            cell_c = int(rel_x // cell_w)
                            cell_r = int(rel_y // cell_h)
                            # Check that cell is not a wall or ghost
                            if self.maze_grid[cell_r][cell_c] not in ("#", "A"):
                                # Set goal to player position
                                self.maze_env.goal = (cell_r, cell_c)
                                self.maze_grid = self.maze_env.render().splitlines()

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
            #self._draw_stats_area()
            self._draw_controls()

            if self.show_settings:
                overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 150))
                self.screen.blit(overlay, (0, 0))
                self.settings_panel.draw_settings_panel()

            pygame.display.flip()

        return None
