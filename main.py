import pygame
#Screen classes obtained from their files to use functions and send parameters
from utils.visualization import StartScreen
from utils.selection_menu import SelectionMenuScreen
from utils.maze_train import MazeTrainingScreen
from utils.archery_train import ArcheryTraningScreen
from utils.conf import Config

def main():

    pygame.init()

    #Define Start screen at the beginning of execution with default parameters
    current_screen = "start"
    config = Config()
    start_screen = StartScreen(config)

    #Pygame loop to keep interaction between screens until the code stops or the loop is broken
    while True:

        #Run Start screen to display
        if current_screen == "start":

            result = start_screen.run()

            #String value returned from the Start screen to move to Selection menu
            if result == "selection_menu":

                #Insert parameters from the start screen to keep consistence in screen size, fonts, settings, etc.
                selection_screen = SelectionMenuScreen(config)

                #Change current screen variable for next loop iteration
                current_screen = "selection"

            #End loop and close the display with other result values
            else:
                break

        #Run Selection screen menu to display
        elif current_screen == "selection":

            result = selection_screen.run()

            #String value returned from Selection screen to return to Start menu
            if result == "start":
                #Set current_screen value for next loop iteration
                current_screen = "start"

                #Creation function must be called again to set the correct screen size when changed in settings
                start_screen = StartScreen(config)

            #String value returned to move to the Maze Training screen
            elif result == "maze_train":
                # Maze Training insance (reuses surface and resources)
                maze_train_screen = MazeTrainingScreen(config)
                #Run the new screen to disaply
                train_result = maze_train_screen.run()

                #Move back to Selection screen by changing the current_screen value
                if train_result in ("selection", "back"):
                    current_screen = "selection"

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(config)
                
                #Returns directly to the Start screen (NOT USED IN CURRENT maze_train CODE)
                elif train_result == "start":
                    current_screen = "start"
                else:
                    #Return to selection screen as default
                    current_screen = "selection"

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(config)

            #String value returned from the selection screen to display the Archery Training screen
            elif result == "archery_train":
                #Archery Training instance with Selection screen parameters and settings
                archery_train_screen = ArcheryTraningScreen(config)

                #Run Archery Training screen to display
                train_result = archery_train_screen.run()

                #Move back to Selection screen from Archery Training with the returned value
                if train_result in ("selection, back"):

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(config)

                #Returns directly to the Start screen (NOT USED IN CURRENT archery_train CODE)
                elif train_result == "start":
                    current_screen = "start"
                
                else:
                    #Return to selection screen as default
                    current_screen = "selection"

                    #Creation function must be called again to set the correct screen size when changed in maze_train settings
                    selection_screen = SelectionMenuScreen(config)


            else:
                break


if __name__ == "__main__":
    main()
