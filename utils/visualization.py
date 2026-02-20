import pygame
import os
import math
from utils.settings_state import SettingsState
from utils.conf import WINDOW_WIDTH, WINDOW_HEIGHT, FPS, FONTS_PATH, IMAGES_PATH, SOUNDS_PATH

# ============================================================
# INTERACTIVE BUTTON CLASS
# ============================================================

class Button:
    def __init__(self, rect, text, font, base_color, hover_color, click_sound=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.current_color = base_color
        self.click_sound = click_sound

        self.scale_factor = 1.0
        self.animation_speed = 0.1

    #Draw the button item with the received values
    def draw(self, screen):
        scaled_rect = self.rect.inflate(
            self.rect.width * (self.scale_factor - 1),
            self.rect.height * (self.scale_factor - 1)
        )

        pygame.draw.rect(screen, self.current_color, scaled_rect)
        pygame.draw.rect(screen, (255, 255, 255), scaled_rect, 3)

        text_surface = self.font.render(self.text, False, (255, 255, 255))
        text_rect = text_surface.get_rect(center=scaled_rect.center)
        screen.blit(text_surface, text_rect)

    #Change button appearance when a mouse hovers over it
    def update(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.hover_color
            self.scale_factor = min(1.08, self.scale_factor + self.animation_speed)
        #Return to original button appearance when the mouse moves away
        else:
            self.current_color = self.base_color
            self.scale_factor = max(1.0, self.scale_factor - self.animation_speed)

    #Play sound effect when the button is clicked
    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.click_sound:
                    self.click_sound.play()
                return True
        return False

#Button class variation that recieves a file path instead of plain text to display an image
class Icon_Button:
    def __init__(self, rect, path, font, base_color, hover_color, click_sound=None):
        self.rect = pygame.Rect(rect)
        self.path = path
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.current_color = base_color
        self.click_sound = click_sound

        self.image = pygame.image.load(path).convert_alpha()

        self.image = pygame.transform.scale(
                self.image,
                (self.rect.width - 20, self.rect.height - 20)
            )

        self.scale_factor = 1.0
        self.animation_speed = 0.1

    #Draw the button item with the received values
    def draw(self, screen):
        scaled_rect = self.rect.inflate(
            self.rect.width * (self.scale_factor - 1),
            self.rect.height * (self.scale_factor - 1)
        )

        pygame.draw.rect(screen, self.current_color, scaled_rect)
        pygame.draw.rect(screen, (255, 255, 255), scaled_rect, 3)

        image_rect = self.image.get_rect(center=self.rect.center)
        screen.blit(self.image, image_rect)

    #Change button appearance when a mouse hovers over it
    def update(self, mouse_pos):
        if self.rect.collidepoint(mouse_pos):
            self.current_color = self.hover_color
            self.scale_factor = min(1.08, self.scale_factor + self.animation_speed)
        #Return to original button appearance when the mouse moves away
        else:
            self.current_color = self.base_color
            self.scale_factor = max(1.0, self.scale_factor - self.animation_speed)

    #Play sound effect when the button is clicked
    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.click_sound:
                    self.click_sound.play()
                return True
        return False


# ============================================================
# SETTINGS PANEL
# ============================================================

class SettingsPanel:
    def __init__(self, screen, settings, click_sound, font_button):
        self.screen = screen
        self.settings = settings
        self.click_sound = click_sound
        self.font_button = font_button

        self.create_settings_buttons()


    #Create a 'back' button which closes the settings panel
    def create_settings_buttons(self):
        width, height = self.screen.get_size()
        center_x = width // 2

        #"BACK" button properties
        self.btn_back = Button(
            rect=(center_x - 100, height // 2 + 140, 200, 50),
            text="BACK",
            font=self.font_button,
            base_color=(60, 60, 90),
            hover_color=(90, 90, 140),
            click_sound=self.click_sound
        )

    def draw_volume_bar(self, center_x, y, value):
        #Draw 10 small squares to represent sound volume in Music and SFX settings 
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
        #Draw arrow buttons to control volume
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
        panel_y = height * 0.20   #🔥
        panel = pygame.Rect(panel_x, panel_y, panel_width, panel_height)

        pygame.draw.rect(self.screen, (20, 20, 30), panel)
        pygame.draw.rect(self.screen, (255, 255, 255), panel, 3)

        # ================= COLUMNS =================
        left_margin = panel_x + 20
        right_margin = panel.right - 60

        # Horizontal starting point for the bars
        bar_center_x = panel_x + panel_width * 0.65

        # ================= VERTICAL LAYOUT =================
        top_padding = 100
        bottom_padding = 80

        usable_height = panel_height - top_padding - bottom_padding
        section_spacing = usable_height // 3

        current_y = panel_y + top_padding

        # ====================================================
        # ================= MUSIC =============================
        # ====================================================

        #Render label for Muusic option
        music_label = self.font_button.render("Music", False, (255, 255, 255))
        music_rect = music_label.get_rect(midleft=(left_margin, current_y))
        self.screen.blit(music_label, music_rect)

        #Draw volume bar and arrows on screen
        left, right = self.draw_volume_bar(bar_center_x, current_y, self.settings.music_volume)
        self.music_arrows = self.draw_arrows(left, right, current_y)

        # ====================================================
        # ================= SFX ===============================
        # ====================================================

        current_y += section_spacing

        #Render label for Sound effects option
        sfx_label = self.font_button.render("SFX", False, (255, 255, 255))
        sfx_rect = sfx_label.get_rect(midleft=(left_margin, current_y))
        self.screen.blit(sfx_label, sfx_rect)

        #Draw volume bar and arrows on screen
        left, right = self.draw_volume_bar(bar_center_x, current_y, self.settings.sfx_volume)
        self.sfx_arrows = self.draw_arrows(left, right, current_y)

        # ====================================================
        # ================= SCREEN MODE =======================
        # ====================================================

        current_y += section_spacing

        #Render label for screen mode option
        mode_label = self.font_button.render("Screen", False, (255, 255, 255))
        mode_rect_label = mode_label.get_rect(midleft=(left_margin, current_y))
        self.screen.blit(mode_label, mode_rect_label)

        #Switches the fullscreen switch option depending on the returned current screen mode
        mode_text = "FULLSCREEN" if self.settings.fullscreen else "WINDOWED"
        mode_surface = self.font_button.render(mode_text, False, (200, 200, 200))

        mode_rect = mode_surface.get_rect(center=(bar_center_x, current_y))
        self.screen.blit(mode_surface, mode_rect)

        #Set separation distance for configuration arrows
        left_x = mode_rect.left - 25
        right_x = mode_rect.right + 25
        self.screen_arrows = self.draw_arrows(left_x, right_x, current_y)

        # ====================================================
        # ================= BACK BUTTON =======================
        # ====================================================

        self.btn_back.rect.center = (panel.centerx, panel.bottom - 50)
        self.btn_back.update(pygame.mouse.get_pos())
        self.btn_back.draw(self.screen)

    def handle_settings_click(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN:
            return

        pos = event.pos

        #Instantly apply settings when volume arrows are clicked to +1 or -1

        #--Music--
        left_m, right_m = self.music_arrows
        if left_m.collidepoint(pos):
            self.settings.music_volume = max(0, self.settings.music_volume - 1)
            self.settings.apply_music_volume()

        if right_m.collidepoint(pos):
            self.settings.music_volume = min(10, self.settings.music_volume + 1)
            self.settings.apply_music_volume()

        #--SFX--
        left_s, right_s = self.sfx_arrows
        if left_s.collidepoint(pos):
            self.settings.sfx_volume = max(0, self.settings.sfx_volume - 1)
            self.settings.apply_sfx_volume(self.click_sound)

        if right_s.collidepoint(pos):
            self.settings.sfx_volume = min(10, self.settings.sfx_volume + 1)
            self.settings.apply_sfx_volume(self.click_sound)

        #Toggle fullscreen/windowed mode
        left_sc, right_sc = self.screen_arrows
        if left_sc.collidepoint(pos) or right_sc.collidepoint(pos):
            self.screen = self.settings.toggle_fullscreen(
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            self.create_settings_buttons()

        # Return whether the back button was clicked
        return self.btn_back.is_clicked(event)


# ============================================================
# START SCREEN
# ============================================================

class StartScreen:

    def __init__(self, screen, fullscreen):
        #Audio config
        self.settings = SettingsState()
        self.clock = pygame.time.Clock()

        if fullscreen != None:
            self.settings.fullscreen = fullscreen
        else:
            self.settings.fullscreen = False

        if screen != None:
            self.screen = screen
        else:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)) #Define default screen mode and size

        pygame.display.set_caption("AI Phantom")

        self.running = True
        self.show_settings = False

        pygame.mixer.init()

        self.load_sounds()
        self.load_music()

        retro_font_path = os.path.join(FONTS_PATH, "RetroGaming.ttf")

        self.font_title = pygame.font.Font(retro_font_path, 100)
        self.font_statsTitle = pygame.font.Font(retro_font_path, 72)
        self.font_button = pygame.font.Font(retro_font_path, 36)

        self.create_buttons()
        self.settings_panel = SettingsPanel(self.screen, self.settings, self.click_sound, self.font_button)

        # ====================================================
        # ANIMATED BACKGROUND
        # ====================================================
        self.bg_time = 0
        self.bg_speed = 0.5      #Animation speed (lower value = slower animation)
        self.bg_amplitude = 25   #Movement amplitude (lower value = smoother movement)
        self.load_background()

    # --------------------------------------------------------

    #Load background image
    def load_background(self):
        bg_path = os.path.join(IMAGES_PATH, "newBackground1.png")
        if os.path.exists(bg_path):
            #Load background image
            self.background_original = pygame.image.load(bg_path).convert()

            width, height = self.screen.get_size()

            #Margin based on amplitude
            extra_height = self.bg_amplitude * 2

            self.background = pygame.transform.scale(
                self.background_original,
                (width, height + extra_height)
            )
        #Define background as None if no image is on the path
        else:
            self.background_original = None
            self.background = None


    # --------------------------------------------------------

    #Draw background image
    def draw_background(self):
        if not self.background_original:
            return

        width, height = self.screen.get_size()

        #Altitude calculated with animation amplitude
        extra_height = self.bg_amplitude * 2

        #Rezising
        if (not hasattr(self, "background") or
            self.background.get_size() != (width, height + extra_height)):

            self.background = pygame.transform.scale(
                self.background_original,
                (width, height + extra_height)
            )

        #Floating movement
        self.bg_time += self.bg_speed
        offset_y = math.sin(self.bg_time * 0.02) * self.bg_amplitude

        #Background is drawn with a compensation for animation amplitude
        self.screen.blit(self.background, (0, offset_y - self.bg_amplitude))

    # --------------------------------------------------------

    def load_sounds(self):
        #Get sound effects from the predefined path on the variable
        click_path = os.path.join(SOUNDS_PATH, "ButtonClick.mp3")
        if os.path.exists(click_path):
            #Apply sound effect on button settings
            self.click_sound = pygame.mixer.Sound(click_path)
            self.settings.apply_sfx_volume(self.click_sound)
        else:
            self.click_sound = None

    # --------------------------------------------------------

    def load_music(self):
        #Get music from the predefined path on the variable
        music_path = os.path.join(SOUNDS_PATH, "MenuTheme.mp3")
        if os.path.exists(music_path):
            #Apply 
            pygame.mixer.music.load(music_path)
            self.settings.apply_music_volume()
            pygame.mixer.music.play(-1)

    # --------------------------------------------------------

    def create_buttons(self):
        width, height = self.screen.get_size()
        center_x = width // 2
        center_y = height // 2

        self.btn_start = Button(
            rect=(center_x - 150, center_y - 80, 300, 60),
            text="START",
            font=self.font_button,
            base_color=(40, 40, 60),
            hover_color=(80, 80, 120),
            click_sound=self.click_sound
        )

        self.btn_settings = Button(
            rect=(center_x - 150, center_y, 300, 60),
            text="SETTINGS",
            font=self.font_button,
            base_color=(40, 40, 60),
            hover_color=(80, 80, 120),
            click_sound=self.click_sound
        )

        self.btn_exit = Button(
            rect=(center_x - 150, center_y + 80, 300, 60),
            text="EXIT",
            font=self.font_button,
            base_color=(120, 40, 40),
            hover_color=(160, 60, 60),
            click_sound=self.click_sound
        )

    # --------------------------------------------------------

    def draw_title(self):
        width, height = self.screen.get_size()
        title_surface = self.font_title.render("AI PHANTOM", False, (255, 255, 255))
        rect = title_surface.get_rect(center=(width // 2, height // 5))
        self.screen.blit(title_surface, rect)

    # --------------------------------------------------------

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            mouse_pos = pygame.mouse.get_pos()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    self.running = False

                if self.show_settings:
                    if self.settings_panel.handle_settings_click(event):
                        self.show_settings = False
                else:
                    if self.btn_start.is_clicked(event):
                        return "selection_menu"

                    if self.btn_settings.is_clicked(event):
                        self.show_settings = True

                    if self.btn_exit.is_clicked(event):
                        self.running = False

            # ====================================================
            # DIBUJADO
            # ====================================================
            self.screen.fill((15, 15, 25))

            # Fondo animado
            self.draw_background()

            self.draw_title()

            if self.show_settings:
                self.settings_panel.draw_settings_panel()
            else:
                self.btn_start.update(mouse_pos)
                self.btn_settings.update(mouse_pos)
                self.btn_exit.update(mouse_pos)

                self.btn_start.draw(self.screen)
                self.btn_settings.draw(self.screen)
                self.btn_exit.draw(self.screen)

            pygame.display.flip()
        
        return None
