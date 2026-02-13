import pygame
import math
import os
from utils.visualization import Button, FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from utils.settings_state import SettingsState


class SelectionMenuScreen:

    def __init__(self, screen, click_sound, font_title, font_statsTitle, font_button, settings: SettingsState):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True

        self.click_sound = click_sound
        self.font_title = font_title
        self.font_statsTitle = font_statsTitle
        self.font_button = font_button

        self.settings = settings

        # ✅ Nuevo flag
        self.show_stats = False

        self.create_top_buttons()
        self.load_card_images()
        self.create_cards()
        self.create_stats_buttons()

        # Background
        self.bg_time = 0
        self.bg_speed = 0.5
        self.bg_amplitude = 25

        self.load_background()

    # --------------------------------------------------------
    # IMAGES
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

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

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
    # TOP BUTTONS
    # --------------------------------------------------------

    def create_top_buttons(self): 
        width, height = self.screen.get_size() 

        self.btn_back_menu = Button(
            rect=(30, 30, 140, 50),
            text="BACK",
            font=self.font_button,
            base_color=(60, 60, 90),
            hover_color=(90, 90, 140),
            click_sound=self.click_sound
        )

        # ✅ CAMBIADO A STATS
        self.btn_stats = Button(
            rect=(width - 280, 30, 250, 50),
            text="STATS",
            font=self.font_button,
            base_color=(40, 40, 60),
            hover_color=(80, 80, 120),
            click_sound=self.click_sound
        )

    # --------------------------------------------------------
    # STATS BUTTONS
    # --------------------------------------------------------

    def create_stats_buttons(self):
        self.btn_back_stats = Button(
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
    # STATS PANEL (NUEVO)
    # --------------------------------------------------------

    def draw_stats_panel(self):

        width, height = self.screen.get_size()
        panel = pygame.Rect(width * 0.15, height * 0.2, width * 0.7, height * 0.65)

        pygame.draw.rect(self.screen, (20, 20, 30), panel)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, 3)

        title = self.font_statsTitle.render("PLAYER STATS", False, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(panel.centerx, panel.top + 60)))

        # Placeholder seguro
        lines = [
            "Games Played: 0",
            "Wins: 0",
            "Best Score: 0",
            "Best Time: --"
        ]

        y = panel.top + 150
        for line in lines:
            txt = self.font_button.render(line, False, (200, 200, 200))
            self.screen.blit(txt, txt.get_rect(center=(panel.centerx, y)))
            y += 50

        self.btn_back_stats.rect.center = (panel.centerx, panel.bottom - 50)
        self.btn_back_stats.update(pygame.mouse.get_pos())
        self.btn_back_stats.draw(self.screen)

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

                if self.show_stats:
                    if self.btn_back_stats.is_clicked(event):
                        self.show_stats = False

                else:
                    if self.btn_back_menu.is_clicked(event):
                        return "start"

                    if self.btn_stats.is_clicked(event):
                        self.show_stats = True

                    if event.type == pygame.MOUSEBUTTONDOWN:
                        for card in self.cards:
                            if card["rect"].collidepoint(event.pos):
                                title = card["title"]
                                if title == "Maze":
                                    return "maze_train"
                                if title == "Archery":
                                    return "archery_train"
                                print(f"Selected: {card['title']}")

            self.screen.fill((18, 18, 28))
            self.draw_background()

            if self.show_stats:
                self.draw_stats_panel()
            else:
                self.btn_back_menu.update(mouse_pos)
                self.btn_stats.update(mouse_pos)

                self.btn_back_menu.draw(self.screen)
                self.btn_stats.draw(self.screen)

                self.draw_cards(mouse_pos)

            pygame.display.flip()

        return None
