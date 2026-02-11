import pygame
from game.ghost import Ghost

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

ghost = Ghost()
ghost_image_key = "idle"
ghost_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        ghost_pos.y -= 300 * dt
        ghost_image_key = "back"
    if keys[pygame.K_s]:
        ghost_pos.y += 300 * dt
        ghost_image_key = "front"
    if keys[pygame.K_a]:
        ghost_pos.x -= 300 * dt
        ghost_image_key = "left"
    if keys[pygame.K_d]:
        ghost_pos.x += 300 * dt
        ghost_image_key = "right"

    ghost.update(ghost_pos, ghost_image_key, screen)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
