import random
import numpy as np

WALL = 1
PATH = 0


def generate_dfs_maze(width: int, height: int, seed: int | None = None,
                      loop_probability: float = 0.05) -> np.ndarray:
    """
    DFS Recursive Backtracker Maze Generator
    0 = PATH, 1 = WALL
    Siempre resoluble.
    """

    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("Maze dimensions must be odd numbers.")

    rng = random.Random(seed)

    grid = [[WALL for _ in range(width)] for _ in range(height)]

    def carve(x, y):
        grid[y][x] = PATH

        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        rng.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                if grid[ny][nx] == WALL:
                    grid[y + dy // 2][x + dx // 2] = PATH
                    carve(nx, ny)

    carve(1, 1)

    # Post-processing loops
    for y in range(1, height - 1):
        for x in range(1, width - 1):

            if grid[y][x] == WALL:

                neighbors = 0
                if grid[y + 1][x] == PATH:
                    neighbors += 1
                if grid[y - 1][x] == PATH:
                    neighbors += 1
                if grid[y][x + 1] == PATH:
                    neighbors += 1
                if grid[y][x - 1] == PATH:
                    neighbors += 1

                if neighbors >= 2 and rng.random() < loop_probability:
                    grid[y][x] = PATH

    return np.array(grid, dtype=np.int8)
