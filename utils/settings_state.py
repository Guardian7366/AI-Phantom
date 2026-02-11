import pygame


class SettingsState:
    """
    Estado global compartido entre pantallas.
    Maneja:
    - Volumen música
    - Volumen efectos
    - Fullscreen
    """

    def __init__(self):
        self.music_volume = 5
        self.sfx_volume = 5
        self.fullscreen = False

    # --------------------------------------------------------

    def apply_music_volume(self):
        pygame.mixer.music.set_volume(self.music_volume / 10)

    # --------------------------------------------------------

    def apply_sfx_volume(self, click_sound):
        if click_sound:
            click_sound.set_volume(self.sfx_volume / 10)

    # --------------------------------------------------------

    def toggle_fullscreen(self, window_size):
        self.fullscreen = not self.fullscreen

        if self.fullscreen:
            return pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            return pygame.display.set_mode(window_size)
