import pygame
import math
import random
import os
import torch

from ai_phantom.agents.ppo.policy import Policy
from ai_phantom.core.checkpointing import load_checkpoint
from ai_phantom.agents.ppo import CnnActorCritic, PPOConfig, PPOTrainer
from ai_phantom.core.device import select_device
from ai_phantom.envs.maze.maze_env import MazeConfig, MazeEnv
from ai_phantom.planners.bfs import bfs_plan, path_to_actions
from utils.start_menu import Icon_Button, SettingsPanel
from utils.conf import WINDOW_WIDTH, FPS, Config, PHASE_0, PHASE_BC, PHASE_1, FINAL_1


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
        self.caught_sound = config.caught_sound
        self.font_title = config.font_title
        self.font_statsTitle = config.font_statsTitle
        self.font_button = config.font_button
        self.font_text = config.font_text
        
        self.running = True

        #Playback state
        self.current_tick = 0

        #Flags
        self.show_settings = False

        self.maze_cfg = MazeConfig(
            height=12,
            width=12,
            use_walls=True,
            max_steps=50,
            min_manhattan=6,
            include_dist_channel=True,
            terminate_on_loop=False,
            loop_terminate_hits=999,
        )
        self.maze_env = MazeEnv(self.maze_cfg, seed=0)
        self.maze_grid = None
        self.maze_actions = None
        self.current_action = "idle"

        #Create buttons
        self._create_buttons()
        #Compute layout
        self._recalc_layout()

        self.settings.apply_music_volume()
        self.settings.apply_sfx_volume(self.click_sound)
        self.settings.apply_sfx_volume(self.caught_sound)

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
        self.cross_img = pygame.image.load("assets/sprites/misc/RedCross.png").convert_alpha()

        self.mouse_in_maze_pos = None

        dev_cfg = select_device(device="auto", allow_tf32=True, cudnn_benchmark=True)
        self.device = dev_cfg.device
        ppo_cfg = PPOConfig()
        obs, _ = self.maze_env.reset(seed=0, phase=1)
        obs_shape = tuple(obs.shape)
        model = CnnActorCritic(obs_shape=obs_shape, num_actions=4)
        trainer = PPOTrainer(model=model, cfg=ppo_cfg, device=self.device)

        self.policy = Policy(
            model=trainer.model,
            enable_action_mask=True,
            nan_repl=float(ppo_cfg.nan_logits_replacement),
            fallback_action=0,
        )

        checkpoint_kwargs = {
            "model": trainer.model,
            "optimizer": trainer.optim,
            "map_location": self.device,
            "restore_rng": False,
        }

        if os.path.exists(FINAL_1):
            load_checkpoint(FINAL_1, **checkpoint_kwargs)
        elif os.path.exists(PHASE_1):
            load_checkpoint(PHASE_1, **checkpoint_kwargs)
        elif os.path.exists(PHASE_BC):
            load_checkpoint(PHASE_BC, **checkpoint_kwargs)
        elif os.path.exists(PHASE_0):
            load_checkpoint(PHASE_0, **checkpoint_kwargs)
        else:
            raise RuntimeError("No training found. Run training first to play the game.")


    # ----------------------
    # CREATION & LAYOUT
    # ----------------------
    def _create_buttons(self):
        #Main buttons
        self.btn_back = Icon_Button((20, 20, 70, 70), "assets/images/back.png", self.font_button, (60, 60, 90), (90, 90, 140), click_sound=self.click_sound)
        self.btn_settings = Icon_Button((WINDOW_WIDTH - 160, 20, 75, 75), "assets/images/gear.png", self.font_button, (40, 40, 60), (80, 80, 120), click_sound=self.click_sound)

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

        self.btn_back.rect.topleft = (20, 30)
        self.btn_back.rect.size = (70, 70)

        self.btn_settings.rect.topright = (width - 20, 20)
        self.btn_settings.rect.size = (75, 75)

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
                    image = self.player_img
                elif char == "A":
                    image = self.ghost_img.get(self.current_action, self.ghost_img["idle"])
                else:
                    image = self.floor_img
                    if self.mouse_in_maze_pos == (r, c):
                        image = self.cross_img

                # Dibujar piso primero
                floor_scaled = pygame.transform.scale(self.floor_img, (rect.width, rect.height))
                self.screen.blit(floor_scaled, rect)
                # Luego dibujar el sprite correspondiente encima
                image = pygame.transform.scale(image, (rect.width, rect.height))
                self.screen.blit(image, rect)

    def _draw_controls(self):
        mouse_pos = pygame.mouse.get_pos()
        #Update/draw top controls
        for b in (self.btn_back, self.btn_settings):
            b.update(mouse_pos)
            b.draw(self.screen)


        #Current Steps Label
        self.steps_title = self.font_button.render("STEPS", False, (230, 230, 230))
        rectGT = self.steps_title.get_rect(center=self.ghost_title_pos)
        self.screen.blit(self.steps_title, rectGT)

        #Current Steps
        self.cur_stepsLabel = self.font_button.render(str(self.maze_env.t), False, (230, 230, 230))
        rectGS = self.cur_stepsLabel.get_rect(center=self.ghost_score_pos)
        self.screen.blit(self.cur_stepsLabel, rectGS)

        #Max Steps Label
        self.limit_title = self.font_button.render("LIMIT", False, (230, 230, 230))
        rectPT = self.limit_title.get_rect(center=self.player_title_pos)
        self.screen.blit(self.limit_title, rectPT)

        #Max Steps
        self.limitLabel = self.font_button.render(str(self.maze_cfg.max_steps), False, (230, 230, 230))
        rectPS = self.limitLabel.get_rect(center=self.player_score_pos)
        self.screen.blit(self.limitLabel, rectPS)

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

        return None

    def move_ghost(self):
        if pygame.time.get_ticks() - self.current_tick > 250:
            self.current_tick = pygame.time.get_ticks()

            obs = self.maze_env._make_obs()
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()

            with torch.no_grad():
                out = self.policy.act(obs_t, deterministic=True)

            action = int(out.action.item())
            self.current_action = action

            obs, _, done, info = self.maze_env.step(action)
            self.maze_grid = self.maze_env.render().splitlines()

            if done:
                self.current_action = "idle"  # Reset action to idle when episode ends
                self.maze_grid = None  # Trigger new maze generation
                if info["reached"]:
                    self.caught_sound.play() #Play sound effect when ghost catches the player
                else:
                    self.maze_cfg.max_steps += 5  # Increase max steps limit if ghost fails to catch player



    # ----------------------------------
    # MAIN LOOP
    # ----------------------------------
    def run(self):
        while self.running:
            self.clock.tick(FPS)
            # Recompute layout only if size changed
            self._recalc_layout()

            if self.maze_grid is None:
                seed = random.randint(0, 9999)
                wall_prob = random.uniform(0.1, 0.35)  # Random wall density for variability
                self.maze_env.rebuild_walls(seed=seed, wall_prob=wall_prob)  # Ensure new maze layout
                obs, info = self.maze_env.reset(seed=seed, phase=1)
                self.maze_env.goal = None  # Clear goal to allow player placement
                self.maze_grid = self.maze_env.render().splitlines()

            if self.maze_env.goal is not None:
                self.move_ghost()  # Start the ghost movement logic

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                if self.maze_env.goal is None and not self.show_settings:
                    # Get mouse position
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
                            # Store mouse position in maze coordinates for feedback
                            self.mouse_in_maze_pos = (cell_r, cell_c)
                            # Set goal to player position when click
                            if event.type == pygame.MOUSEBUTTONDOWN:
                                self.maze_env.goal = (cell_r, cell_c)
                                self.maze_grid = self.maze_env.render().splitlines()
                                self.mouse_in_maze_pos = None  # Clear stored position after setting goal
                        else:
                            self.mouse_in_maze_pos = None  # Clear stored position if hovering over invalid cell
                    else:
                        self.mouse_in_maze_pos = None # Clear stored position when mouse leaves maze area

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
