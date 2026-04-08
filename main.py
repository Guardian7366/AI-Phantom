import pygame
#Screen classes obtained from their files to use functions and send parameters
from utils.start_menu import StartScreen
from utils.selection_menu import SelectionMenuScreen
from utils.maze_train import MazeTrainingScreen
from utils.maze_game import MazeGameScreen
from utils.conf import Config

""" JuanS si participó pero al haber un error de configuración con el correo no sale como contribuidor, sin embargo
cuenta con testigos de su participación activa en el proyecto"""

def main():

    pygame.init()

    #Define Start screen at the beginning of execution with default parameters
    prev_screen_name = "start"
    new_screen_name = "start"
    config = Config()

    current_screen = StartScreen(config)
    config.play_menu_music()

    #Pygame loop to keep interaction between screens until the code stops or the loop is broken
    while True:
        new_screen_name = current_screen.run()

        if new_screen_name == "start":
            current_screen = StartScreen(config)
            prev_screen_name = "start"
        elif new_screen_name == "selection":
            current_screen = SelectionMenuScreen(config)
            if prev_screen_name != "start":
                config.play_menu_music() 
            prev_screen_name = "selection"
        elif new_screen_name == "maze_train":
            current_screen = MazeTrainingScreen(config, show_start_popup=(prev_screen_name != "maze_train"))
            prev_screen_name = "maze_train"
            config.play_maze_music()
        elif new_screen_name == "maze_game":
            current_screen = MazeGameScreen(config)
            config.play_maze_music()
            prev_screen_name = "maze_game"
        else:
            break


if __name__ == "__main__":
    main()
