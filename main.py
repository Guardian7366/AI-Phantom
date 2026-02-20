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
    current_screen_name = "start"
    new_screen_name = current_screen_name
    config = Config()

    current_screen = StartScreen(config)

    #Pygame loop to keep interaction between screens until the code stops or the loop is broken
    while True:
        new_screen_name = current_screen.run()

        if new_screen_name != current_screen_name:
            if new_screen_name == "start":
                current_screen = StartScreen(config)
            elif new_screen_name == "selection":
                current_screen = SelectionMenuScreen(config)
            elif new_screen_name == "maze_train":
                current_screen = MazeTrainingScreen(config)
            elif new_screen_name == "archery_train":
                current_screen = ArcheryTraningScreen(config)
            else:
                break
            current_screen_name = new_screen_name


if __name__ == "__main__":
    main()
