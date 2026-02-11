import sys
import pygame

from utils.visualization import StartScreen


def main():
    pygame.init()

    app = StartScreen()
    app.run()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
