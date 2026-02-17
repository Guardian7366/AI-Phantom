# utils/maze_train.py
import pygame
import math
import os
from typing import Optional

from utils.visualization import Button, Icon_Button, WINDOW_WIDTH, WINDOW_HEIGHT, FPS
from utils.settings_state import SettingsState

# Flags para reducir tartamudeo al toggle de fullscreen (opcional)
FULLSCREEN_FLAGS = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
WINDOWED_FLAGS = pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF

ASSETS_PATH = "assets"
FONTS_PATH = os.path.join(ASSETS_PATH, "fonts")


class MazeTrainingScreen:
    """
    Pantalla de entrenamiento del laberinto (UI).
    - Compatible con las fuentes y tamaños de StartScreen.
    - Usa SettingsState para sincronizar volúmenes y fullscreen.
    """

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

        # NUEVO: botón BACK específico para el overlay de settings
        # Lo creamos aquí pero su posición la actualizaremos cuando dibujemos el panel
        self.btn_back_overlay = Button((0, 0, 200, 50), "BACK", self.font_button, (60, 60, 90), (90, 90, 140), click_sound=self.click_sound)

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
    # NUEVO: dibujo de barra de volumen + flechas
    # ----------------------
    def draw_volume_bar(self, center_x, y, value):
        """
        Dibuja 10 cuadritos centrados en center_x, en la coordenada y.
        Retorna (start_x, end_x) — coordenadas horizontales de inicio y fin (incluyendo ancho total).
        """
        block_size = 22
        spacing = 6
        total_width = 10 * block_size + 9 * spacing

        start_x = int(center_x - total_width // 2)
        for i in range(10):
            rect = pygame.Rect(
                start_x + i * (block_size + spacing),
                y,
                block_size,
                block_size
            )

            if i < value:
                pygame.draw.rect(self.screen, (100, 200, 100), rect, border_radius=4)
            else:
                pygame.draw.rect(self.screen, (60, 60, 60), rect, border_radius=4)

        return start_x, start_x + total_width

    def draw_arrows(self, left_x, right_x, y):
        """
        Dibuja rectángulos con '<' y '>' a los lados y devuelve (left_rect, right_rect)
        left_x = coordenada X del inicio de la barra; right_x = X final de la barra
        y = coordenada vertical (top) donde la barra está dibujada
        """
        size = 32
        left_rect = pygame.Rect(left_x - size - 15, y - 5, size, size)
        right_rect = pygame.Rect(right_x + 15, y - 5, size, size)

        pygame.draw.rect(self.screen, (80, 80, 120), left_rect, border_radius=6)
        pygame.draw.rect(self.screen, (80, 80, 120), right_rect, border_radius=6)

        left_text = self.font_button.render("<", True, (255, 255, 255))
        right_text = self.font_button.render(">", True, (255, 255, 255))

        self.screen.blit(left_text, left_text.get_rect(center=left_rect.center))
        self.screen.blit(right_text, right_text.get_rect(center=right_rect.center))

        return left_rect, right_rect

    def draw_settings_panel(self):
        width, height = self.screen.get_size()

        # ================= PANEL SIZE =================
        panel_width = width * 0.6
        panel_height = height * 0.6

        panel_x = (width - panel_width) // 2
        panel_y = height * 0.20   # 🔥 MÁS ABAJO (ya no tapa el título)

        panel = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        pygame.draw.rect(self.screen, (20, 20, 30), panel)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, 3)

        # ================= COLUMNAS =================
        left_margin = panel_x + 20
        right_margin = panel.right - 60

        # Zona donde comenzarán las barras (MUCHO más a la derecha)
        bar_center_x = panel_x + panel_width * 0.65

        # ================= LAYOUT VERTICAL =================
        top_padding = 100
        bottom_padding = 80

        usable_height = panel_height - top_padding - bottom_padding
        section_spacing = usable_height // 3

        current_y = panel_y + top_padding

        # ====================================================
        # ================= MUSIC =============================
        # ====================================================

        music_label = self.font_button.render("Music", False, (255, 255, 255))
        music_rect = music_label.get_rect(midleft=(left_margin, current_y))
        self.screen.blit(music_label, music_rect)

        left, right = self.draw_volume_bar(bar_center_x, current_y, self.settings.music_volume)
        self.music_arrows = self.draw_arrows(left, right, current_y)

        # ====================================================
        # ================= SFX ===============================
        # ====================================================

        current_y += section_spacing

        sfx_label = self.font_button.render("SFX", False, (255, 255, 255))
        sfx_rect = sfx_label.get_rect(midleft=(left_margin, current_y))
        self.screen.blit(sfx_label, sfx_rect)

        left, right = self.draw_volume_bar(bar_center_x, current_y, self.settings.sfx_volume)
        self.sfx_arrows = self.draw_arrows(left, right, current_y)

        # ====================================================
        # ================= SCREEN MODE =======================
        # ====================================================

        current_y += section_spacing

        mode_label = self.font_button.render("Screen", False, (255, 255, 255))
        mode_rect_label = mode_label.get_rect(midleft=(left_margin, current_y))
        self.screen.blit(mode_label, mode_rect_label)

        mode_text = "FULLSCREEN" if self.settings.fullscreen else "WINDOWED"
        mode_surface = self.font_button.render(mode_text, False, (200, 200, 200))

        mode_rect = mode_surface.get_rect(center=(bar_center_x, current_y))
        self.screen.blit(mode_surface, mode_rect)

        # Flechas con separación real
        left_x = mode_rect.left - 25
        right_x = mode_rect.right + 25
        self.screen_arrows = self.draw_arrows(left_x, right_x, current_y)

        # ====================================================
        # ================= BACK BUTTON =======================
        # ====================================================

        self.btn_back_overlay.rect.center = (panel.centerx, panel.bottom - 50)
        self.btn_back_overlay.update(pygame.mouse.get_pos())
        self.btn_back_overlay.draw(self.screen)

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

    def handle_settings_click(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN:
            return

        pos = event.pos

        left_m, right_m = self.music_arrows
        if left_m.collidepoint(pos):
            self.settings.music_volume = max(0, self.settings.music_volume - 1)
            self.settings.apply_music_volume()

        if right_m.collidepoint(pos):
            self.settings.music_volume = min(10, self.settings.music_volume + 1)
            self.settings.apply_music_volume()

        left_s, right_s = self.sfx_arrows
        if left_s.collidepoint(pos):
            self.settings.sfx_volume = max(0, self.settings.sfx_volume - 1)
            self.settings.apply_sfx_volume(self.click_sound)

        if right_s.collidepoint(pos):
            self.settings.sfx_volume = min(10, self.settings.sfx_volume + 1)
            self.settings.apply_sfx_volume(self.click_sound)

        left_sc, right_sc = self.screen_arrows
        if left_sc.collidepoint(pos) or right_sc.collidepoint(pos):
            self.screen = self.settings.toggle_fullscreen(
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            self._create_buttons()

        if self.btn_back_overlay.is_clicked(event):
            self.show_settings = False

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
                        self.handle_settings_click(event)
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
                self.draw_settings_panel()

            pygame.display.flip()

        return None
