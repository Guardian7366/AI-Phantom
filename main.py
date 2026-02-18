import pygame
from utils.visualization import StartScreen
from utils.selection_menu import SelectionMenuScreen
from utils.maze_train import MazeTrainingScreen
from utils.archery_train import ArcheryTraningScreen

def main():

    pygame.init()

    current_screen = "start"
    start_screen = StartScreen(None, False)

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

                #Creation function must be called again to set the correct screen size when changed in settings
                start_screen = StartScreen(
                    selection_screen.screen,
                    selection_screen.settings.fullscreen #Boolean value to define if the game is currently windowed or fullscreen
                )

            elif result == "maze_train":
                # Maze Training insance (reuses surface and resources)
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

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(
                    maze_train_screen.screen,
                    maze_train_screen.click_sound,
                    maze_train_screen.font_title,
                    maze_train_screen.font_statsTitle,
                    maze_train_screen.font_button,
                    maze_train_screen.settings
                )
                    
                elif train_result == "start":
                    current_screen = "start"
                else:
                    #Return to selection screen as default
                    current_screen = "selection"

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(
                    maze_train_screen.screen,
                    maze_train_screen.click_sound,
                    maze_train_screen.font_title,
                    maze_train_screen.font_statsTitle,
                    maze_train_screen.font_button,
                    maze_train_screen.settings
                )

            elif result == "archery_train":
                archery_train_screen = ArcheryTraningScreen(
                    screen=selection_screen.screen,
                    settings=selection_screen.settings,
                    click_sound=selection_screen.click_sound,
                    font_title=selection_screen.font_title,
                    font_statsTitle=selection_screen.font_statsTitle,
                    font_button=selection_screen.font_button,
                )

                train_result = archery_train_screen.run()

                if train_result in ("selection, back"):

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(
                    archery_train_screen.screen,
                    archery_train_screen.click_sound,
                    archery_train_screen.font_title,
                    archery_train_screen.font_statsTitle,
                    archery_train_screen.font_button,
                    archery_train_screen.settings
                    )

                elif train_result == "start":
                    current_screen = "start"
                
                else:
                    #Return to selection screen as default
                    current_screen = "selection"

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(
                    archery_train_screen.screen,
                    archery_train_screen.click_sound,
                    archery_train_screen.font_title,
                    archery_train_screen.font_statsTitle,
                    archery_train_screen.font_button,
                    archery_train_screen.settings
                )


            else:
                break


if __name__ == "__main__":
    main()
