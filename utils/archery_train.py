# utils/maze_train.py
import pygame
import math
import os
from typing import Optional

from utils.visualization import Button, Icon_Button, SettingsPanel
from utils.settings_state import SettingsState
from utils.conf import FONTS_PATH, WINDOW_WIDTH, WINDOW_HEIGHT, FPS

class ArcheryTraningScreen:
    def __init__(
        self,
        screen: pygame.Surface,
        settings: SettingsState,
        click_sound: Optional[pygame.mixer.Sound],
        font_title,
        font_statsTitle,
        font_button,
):
        retro_font_path = os.path.join(FONTS_PATH, "RetroGaming.ttf")

        self.screen = screen
        self.settings = settings
        self.click_sound = click_sound
        self.font_title = font_title
        self.font_statsTitle = font_statsTitle
        self.font_button = font_button
        self.font_text = pygame.font.Font(retro_font_path, 20)

        self.clock = pygame.time.Clock()
        self.running = True

        # playback state
        self.playing = False
        self.speed_index = 0
        self.speeds = [1, 2, 4]  # x1, x2, x4

        # flags
        self.show_settings = False

        # create buttons (own instances, no reuse)
        self._create_buttons()
        # compute layout
        self._recalc_layout()

        # apply shared settings
        self.settings.apply_music_volume()
        self.settings.apply_sfx_volume(self.click_sound)

        # storage for arrow rects used in settings overlay
        self._settings_arrow_rects = {}
        self.settings_panel = SettingsPanel(self.screen, self.settings, self.click_sound, self.font_button)

    # ----------------------
    # CREACIÓN & LAYOUT
    # ----------------------
    def _create_buttons(self):
        # sizes consistent with StartScreen
        self.btn_back = Button((20, 20, 140, 50), "BACK", self.font_button, (60, 60, 90), (90, 90, 140), click_sound=self.click_sound)
        self.btn_settings = Icon_Button((WINDOW_WIDTH - 160, 20, 75, 75), "assets/images/gear.png", self.font_button, (40, 40, 60), (80, 80, 120), click_sound=self.click_sound)
        # play bbutton
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

        # rects
        self.maze_rect = pygame.Rect(margin, area_top, maze_w, area_h)
        self.stats_rect = pygame.Rect(margin + maze_w + gap, area_top, stats_w, area_h)

        # title center
        self.title_pos = (width // 2, 48)

        # control positions
        ctl_y = area_top + 10
        play_w, play_h = 170, 44
        ff_w, ff_h = 120, 44

        play_x = self.maze_rect.left + 12
        ff_x = play_x + play_w + 14

        self.btn_play.rect.topleft = (play_x, ctl_y)
        self.btn_play.rect.size = (play_w, play_h)

        self.btn_ff.rect.topleft = (ff_x, ctl_y)
        self.btn_ff.rect.size = (ff_w, ff_h)

        self.btn_back.rect.topleft = (20, 30)
        self.btn_back.rect.size = (140, 50)

        self.btn_settings.rect.topright = (width - 20, 20)
        self.btn_settings.rect.size = (75, 75)

        # btn_back_overlay will be positioned inside the overlay panel when drawing;

    # ----------------------
    # DIBUJADO
    # ----------------------
    def _draw_title(self):
        title = self.font_title.render("TRAINING", False, (255, 255, 255))
        rect = title.get_rect(center=self.title_pos)
        self.screen.blit(title, rect)

    def _draw_maze_area(self):
        pygame.draw.rect(self.screen, (12, 14, 22), self.maze_rect)
        pygame.draw.rect(self.screen, (90, 90, 90), self.maze_rect, 3)

        # grid placeholder using cell size ~32 (similar feel a tu StartScreen grid)
        cols = max(6, self.maze_rect.width // 32)
        rows = max(6, self.maze_rect.height // 32)
        cell_w = self.maze_rect.width / cols
        cell_h = self.maze_rect.height / rows

        for r in range(rows):
            for c in range(cols):
                x = int(self.maze_rect.left + c * cell_w)
                y = int(self.maze_rect.top + r * cell_h)
                rect = pygame.Rect(x, y, math.ceil(cell_w), math.ceil(cell_h))
                color = (18, 20, 30) if (r + c) % 2 == 0 else (14, 16, 24)
                pygame.draw.rect(self.screen, color, rect)

        # border & placeholder text
        pygame.draw.rect(self.screen, (150, 150, 150), self.maze_rect, 2)
        hint = self.font_button.render("Training placeholder", False, (180, 180, 180))
        self.screen.blit(hint, (self.maze_rect.left + 12, self.maze_rect.top + 12))

    def _draw_stats_area(self):
        pygame.draw.rect(self.screen, (16, 18, 26), self.stats_rect)
        pygame.draw.rect(self.screen, (120, 120, 120), self.stats_rect, 2)

        hdr = self.font_text.render("Statistics", False, (230, 230, 230))
        self.screen.blit(hdr, (self.stats_rect.left + 12, self.stats_rect.top + 12))

        y = self.stats_rect.top + 64
        lines = [
            f"Playing: {'Yes' if self.playing else 'No'}",
            f"Speed: x{self.speeds[self.speed_index]}",
            "Episodes: 0",
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
        # update/draw top controls
        for b in (self.btn_back, self.btn_settings, self.btn_play, self.btn_ff):
            b.update(mouse_pos)
            b.draw(self.screen)

        # synchronize labels
        self.btn_play.text = "PAUSE" if self.playing else "PLAY"
        self.btn_ff.text = f"x{self.speeds[self.speed_index]}"

    # ----------------------
    # INTERACCIÓN
    # ----------------------
    def _handle_control_click(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN:
            return None
        pos = event.pos

        if self.btn_back.is_clicked(event):
            # top-left back -> regresar a selection
            return "selection"

        if self.btn_settings.is_clicked(event):
            self.show_settings = True
            return None

        if self.btn_play.is_clicked(event):
            self.playing = not self.playing
            return None

        if self.btn_ff.is_clicked(event):
            self.speed_index = (self.speed_index + 1) % len(self.speeds)
            return None

        return None

    # ----------------------
    # LOOP PRINCIPAL
    # ----------------------
    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            # recompute layout only if size changed (cheap check)
            self._recalc_layout()

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
                    # interaction
                    res = self._handle_control_click(event)
                    if res == "selection":
                        return "selection"

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        return "selection"

                    # clicks inside canvas or stats (placeholder feedback)
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if self.maze_rect.collidepoint(event.pos) or self.stats_rect.collidepoint(event.pos):
                            if self.click_sound:
                                self.click_sound.play()

            # update training logic when playing (placeholder)
            if self.playing:
                # here goes stepping training/generator logic, respecting self.speeds[self.speed_index]
                pass

            # draw
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
