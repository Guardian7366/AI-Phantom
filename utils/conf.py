import os

import pygame

from utils.settings_state import SettingsState

# ============================================================
# BASIC CONFIGURATION
# ============================================================

#Screen dimensions
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 640
FPS = 60

#File paths set into variables to retrieve assets
ASSETS_PATH = "assets"
SOUNDS_PATH = os.path.join(ASSETS_PATH, "sounds")
IMAGES_PATH = os.path.join(ASSETS_PATH, "images")
FONTS_PATH = os.path.join(ASSETS_PATH, "fonts")

# ============================================================

class Config:
    def __init__(self, screen=None, fullscreen=False):
        self.settings = SettingsState()
        self.clock = pygame.time.Clock()

        if fullscreen != None:
            self.settings.fullscreen = fullscreen
        else:
            self.settings.fullscreen = False

        if screen != None:
            self.screen = screen
        else:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)) # Define default screen mode and size

        pygame.display.set_caption("AI Phantom")

        retro_font_path = os.path.join(FONTS_PATH, "RetroGaming.ttf")

        self.font_title = pygame.font.Font(retro_font_path, 100)
        self.font_statsTitle = pygame.font.Font(retro_font_path, 72)
        self.font_button = pygame.font.Font(retro_font_path, 36)
        self.font_text = pygame.font.Font(retro_font_path, 20)

        pygame.mixer.init()

        click_path = os.path.join(SOUNDS_PATH, "ButtonClick.mp3")
        if os.path.exists(click_path):
            self.click_sound = pygame.mixer.Sound(click_path)
            self.settings.apply_sfx_volume(self.click_sound)
        else:
            self.click_sound = None

        music_path = os.path.join(SOUNDS_PATH, "MenuTheme.mp3")
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            self.settings.apply_music_volume()
            pygame.mixer.music.play(-1)
