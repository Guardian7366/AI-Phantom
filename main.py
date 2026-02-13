import pygame
from utils.visualization import StartScreen
from utils.selection_menu import SelectionMenuScreen
from utils.maze_train import MazeTrainingScreen

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
                    start_screen.font_statsTitle,
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

            elif result == "maze_train":
                # instancia la pantalla de entrenamiento (reutilizando la misma surface y recursos)
                maze_train_screen = MazeTrainingScreen(
                    screen=selection_screen.screen,
                    settings=selection_screen.settings,
                    click_sound=selection_screen.click_sound,
                    font_title=selection_screen.font_title,
                    font_statsTitle=selection_screen.font_statsTitle,
                    font_button=selection_screen.font_button,
                )

                train_result = maze_train_screen.run()

                if train_result in ("selection", "back"):
                    current_screen = "selection"
                elif train_result == "start":
                    current_screen = "start"
                else:
                    # por defecto volver al menu de selección
                    current_screen = "selection"

            else:
                break


if __name__ == "__main__":
    main()
