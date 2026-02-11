import pygame


class Ghost(pygame.sprite.Sprite):
    def __init__(self, *groups):
        super().__init__(*groups)
        self.images = {}
        self.images["front"] = pygame.image.load(
            "assets/sprites/phantom/PhantomFront.png"
        )
        self.images["back"] = pygame.image.load(
            "assets/sprites/phantom/PhantomBack.png"
        )
        self.images["left"] = pygame.image.load(
            "assets/sprites/phantom/PhantomLeft.png"
        )
        self.images["right"] = pygame.image.load(
            "assets/sprites/phantom/PhantomRight.png"
        )
        self.images["idle"] = pygame.image.load(
            "assets/sprites/phantom/PhantomIdle.png"
        )
        for key in self.images:
            self.images[key] = pygame.transform.scale(self.images[key], (64, 64))

    def update(self, pos, image_key, screen):
        self.image = self.images.get(image_key, "idle")
        self.rect = self.image.get_rect(center=pos)
        screen.blit(self.image, self.rect)
