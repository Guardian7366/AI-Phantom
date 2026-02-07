from typing import Tuple, Dict, Any, Callable
import numpy as np


class MazeEnvironment:
    """
    Entorno de laberinto compatible con DQN.
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(self, config: dict):
        # Guardar config (IMPORTANTE para evaluación)
        self.config = config

        self.grid = np.array(config["grid"])
        self.start = tuple(config["start"])
        self.goal = tuple(config["goal"])

        self.height, self.width = self.grid.shape

        self.action_space_n = 4
        self.state_dim = 6  # definido previamente en diseño

        self.max_steps = config.get("max_steps", 500)

        self.agent_pos = None
        self.steps = 0

        # Factory para recrear el entorno (clave para evaluación)
        self.factory: Callable[[], "MazeEnvironment"] = (
            lambda: MazeEnvironment(self.config)
        )

    # ---------------------
    # Core API
    # ---------------------

    def reset(self) -> np.ndarray:
        self.agent_pos = list(self.start)
        self.steps = 0
        return self._get_state()

    def step(self, action: int):
        self.steps += 1

        dx, dy = self.ACTIONS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = -0.01
        done = False
        info = {"success": False}

        if self._is_wall(nx, ny):
            reward = -0.1
        else:
            self.agent_pos = [nx, ny]

        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True
            info["success"] = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, info

    # ---------------------
    # Helpers
    # ---------------------

    def _get_state(self) -> np.ndarray:
        ax, ay = self.agent_pos
        gx, gy = self.goal

        return np.array(
            [
                ax / self.height,
                ay / self.width,
                (gx - ax) / self.height,
                (gy - ay) / self.width,
                self._is_wall(ax - 1, ay),
                self._is_wall(ax + 1, ay),
            ],
            dtype=np.float32,
        )

    def _is_wall(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return self.grid[x, y] == 1

