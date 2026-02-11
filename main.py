import pygame
from utils.visualization import StartScreen
from utils.selection_menu import SelectionMenuScreen

def main():

    pygame.init()

    current_screen = "start"
    start_screen = StartScreen()

    while True:

        if current_screen == "start":

            result = start_screen.run()

            if result == "selection_menu":

                selection_screen = SelectionMenuScreen(
                    start_screen.screen,
                    start_screen.click_sound,
                    start_screen.font_title,
                    start_screen.font_button,
                    start_screen.settings
                )


                current_screen = "selection"

            else:
                break

        elif current_screen == "selection":

            result = selection_screen.run()

            if result == "start":
                current_screen = "start"
            else:
                break


if __name__ == "__main__":
    main()
