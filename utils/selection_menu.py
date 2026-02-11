import pygame
import math
import os
from utils.visualization import Button, FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from utils.settings_state import SettingsState


class SelectionMenuScreen:

    def __init__(self, screen, click_sound, font_title, font_button, settings: SettingsState):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True

        self.click_sound = click_sound
        self.font_title = font_title
        self.font_button = font_button

        self.settings = settings
        self.show_settings = False

        self.create_top_buttons()
        self.load_card_images()
        self.create_cards()
        self.create_settings_buttons()
        
        # ====================================================
        # FONDO ANIMADO (MISMO SISTEMA PROTEGIDO)
        # ====================================================
        self.bg_time = 0
        self.bg_speed = 0.5
        self.bg_amplitude = 25

        self.load_background()
        

    # --------------------------------------------------------
    # CARGA IMÁGENES DE LAS CARDS
    # --------------------------------------------------------

    def load_card_images(self):
        image_folder = os.path.join("assets", "images")

        image_names = [
            "mazeCardBackTrain.png",
            "archeryCardBackTrain.png",
            "mazeCardBackHuman.png",
            "archeryCardBackHuman.png"
        ]

        self.card_images_original = []

        for name in image_names:
            path = os.path.join(image_folder, name)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
            else:
                img = None

            self.card_images_original.append(img)

    # ========================================================
    # BACKGROUND SYSTEM (ANTI-VACÍO VISUAL)
    # ========================================================


    def load_background(self):
        bg_path = os.path.join("assets", "images", "newBackground1.png")

        if os.path.exists(bg_path):
            self.background_original = pygame.image.load(bg_path).convert()
        else:
            self.background_original = None

        self.background = None

    def draw_background(self):
        if not self.background_original:
            return

        width, height = self.screen.get_size()

        extra_height = self.bg_amplitude * 2

        if (
            self.background is None
            or self.background.get_size() != (width, height + extra_height)
        ):
            self.background = pygame.transform.scale(
                self.background_original,
                (width, height + extra_height)
            )

        self.bg_time += self.bg_speed
        offset_y = math.sin(self.bg_time * 0.02) * self.bg_amplitude

        self.screen.blit(self.background, (0, offset_y - self.bg_amplitude))

    # --------------------------------------------------------
    # BOTONES SUPERIORES
    # --------------------------------------------------------

    def create_top_buttons(self): 
        width, height = self.screen.get_size() 
        self.btn_back_menu = Button( 
        rect=(30, 30, 140, 50), 
        text="BACK", 
        font=self.font_button, 
        base_color=(60, 60, 90), 
        hover_color=(90, 90, 140), 
        click_sound=self.click_sound)

        self.btn_settings = Button( 
        rect=(width - 280, 30, 250, 50), 
        text="SETTINGS", 
        font=self.font_button, 
        base_color=(40, 40, 60), 
        hover_color=(80, 80, 120), 
        click_sound=self.click_sound)
        
    # --------------------------------------------------------
    # BOTÓN BACK DE SETTINGS
    # --------------------------------------------------------

    def create_settings_buttons(self):
        self.btn_back_settings = Button(
            rect=(0, 0, 180, 55),
            text="BACK",
            font=self.font_button,
            base_color=(60, 60, 90),
            hover_color=(90, 90, 140),
            click_sound=self.click_sound
        )

    # --------------------------------------------------------
    # CARDS
    # --------------------------------------------------------

    def create_cards(self):
        width, height = self.screen.get_size()

        grid_width = width * 0.8
        grid_height = height * 0.65

        start_x = (width - grid_width) // 2
        start_y = height * 0.18

        card_width = grid_width // 2 - 30
        card_height = grid_height // 2 - 30

        self.cards = []

        titles = ["Maze", "Archery", "Coming Soon", "Coming Soon"]

        for row in range(2):
            for col in range(2):
                x = start_x + col * (card_width + 30)
                y = start_y + row * (card_height + 30)

                rect = pygame.Rect(x, y, card_width, card_height)

                index = row * 2 + col

                self.cards.append({
                    "rect": rect,
                    "title": titles[index],
                    "scale": 1.0,
                    "image": self.card_images_original[index]
                })

    def draw_cards(self, mouse_pos):
        for card in self.cards:

            rect = card["rect"]

            if rect.collidepoint(mouse_pos):
                card["scale"] = min(1.05, card["scale"] + 0.05)
            else:
                card["scale"] = max(1.0, card["scale"] - 0.05)

            scaled_rect = rect.inflate(
                rect.width * (card["scale"] - 1),
                rect.height * (card["scale"] - 1)
            )

            if card["image"]:
                scaled_image = pygame.transform.scale(
                    card["image"],
                    (scaled_rect.width, scaled_rect.height)
                )
                self.screen.blit(scaled_image, scaled_rect)
            else:
                pygame.draw.rect(self.screen, (30, 30, 45), scaled_rect)

            pygame.draw.rect(self.screen, (255, 255, 255), scaled_rect, 3)

            title_surface = self.font_button.render(card["title"], False, (255, 255, 255))
            title_rect = title_surface.get_rect(
                center=(scaled_rect.centerx, scaled_rect.bottom - 25)
            )

            self.screen.blit(title_surface, title_rect)

    # --------------------------------------------------------
    # SETTINGS PANEL
    # --------------------------------------------------------

    def draw_volume_bar(self, center_x, y, value):
        block_size = 22
        spacing = 6
        total_width = 10 * block_size + 9 * spacing

        start_x = center_x - total_width // 2

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
        panel = pygame.Rect(width * 0.15, height * 0.2, width * 0.7, height * 0.65)


        pygame.draw.rect(self.screen, (20, 20, 30), panel)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, 3)

        # ==========================================
        # LAYOUT HORIZONTAL EN COLUMNAS
        # ==========================================
        left_column_x = panel.left + 50

        # ancho real de la barra (igual al que usa draw_volume_bar)
        block_size = 22
        spacing = 6
        bar_width = 10 * block_size + 9 * spacing  # 274px

        # ahora centramos la barra en la mitad derecha del panel
        right_half_center = panel.left + panel.width * 0.65
        bar_column_x = right_half_center


        top_margin = 80
        section_spacing = 100

        current_y = panel.top + top_margin

        # ================= MUSIC =================
        music_label = self.font_button.render("Music", False, (255, 255, 255))
        self.screen.blit(
            music_label,
            music_label.get_rect(midleft=(left_column_x, current_y))
        )

        left, right = self.draw_volume_bar(bar_column_x, current_y, self.settings.music_volume)
        self.music_arrows = self.draw_arrows(left, right, current_y)

        # ================= SFX =================
        current_y += section_spacing

        sfx_label = self.font_button.render("SFX", False, (255, 255, 255))
        self.screen.blit(
            sfx_label,
            sfx_label.get_rect(midleft=(left_column_x, current_y))
        )

        left, right = self.draw_volume_bar(bar_column_x, current_y, self.settings.sfx_volume)
        self.sfx_arrows = self.draw_arrows(left, right, current_y)

        # ================= SCREEN =================
        current_y += section_spacing

        mode_label = self.font_button.render("Screen", False, (255, 255, 255))
        self.screen.blit(
            mode_label,
            mode_label.get_rect(midleft=(left_column_x, current_y))
        )

        mode_text = "FULLSCREEN" if self.settings.fullscreen else "WINDOWED"
        mode_surface = self.font_button.render(mode_text, False, (200, 200, 200))
        mode_rect = mode_surface.get_rect(center=(bar_column_x, current_y))
        self.screen.blit(mode_surface, mode_rect)

        left_x = mode_rect.left - 35
        right_x = mode_rect.right + 35
        self.screen_arrows = self.draw_arrows(left_x, right_x, current_y)

        # ================= BACK BUTTON =================
        self.btn_back_settings.rect.center = (
            panel.centerx,
            panel.bottom - 50
        )

        self.btn_back_settings.update(pygame.mouse.get_pos())
        self.btn_back_settings.draw(self.screen)




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
                self.screen,
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            self.create_top_buttons()
            self.create_cards()

        if self.btn_back_settings.is_clicked(event):
            self.show_settings = False

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    def run(self):

        while self.running:
            self.clock.tick(FPS)
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                if self.show_settings:
                    self.handle_settings_click(event)
                else:
                    if self.btn_back_menu.is_clicked(event):
                        return "start"

                    if self.btn_settings.is_clicked(event):
                        self.show_settings = True

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        for card in self.cards:
                            if card["rect"].collidepoint(event.pos):
                                print(f"Selected: {card['title']}")

            self.screen.fill((18, 18, 28))

            # 🔥 Fondo animado protegido
            self.draw_background()

            if self.show_settings:
                self.draw_settings_panel()
            else:
                self.btn_back_menu.update(mouse_pos)
                self.btn_settings.update(mouse_pos)

                self.btn_back_menu.draw(self.screen)
                self.btn_settings.draw(self.screen)

                self.draw_cards(mouse_pos)

            pygame.display.flip()

        return None
