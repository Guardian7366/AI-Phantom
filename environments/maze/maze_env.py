from typing import Callable
import numpy as np


class MazeEnvironment:
    """
    Entorno de laberinto compatible con DQN.
    Acepta:
    - config completo (con clave 'environment')
    - o config directo del entorno
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(self, config: dict):
        # Guardar config completo (para factory / evaluación)
        self.config = config

        # 🔑 Extraer subconfig del entorno de forma segura
        if "environment" in config:
            env_cfg = config["environment"]
        else:
            env_cfg = config

        # ---------------------
        # Validaciones claras
        # ---------------------
        required_keys = ["grid", "start", "goal"]
        for key in required_keys:
            if key not in env_cfg:
                raise KeyError(
                    f"MazeEnvironment: falta clave '{key}' en config['environment']"
                )

        # ---------------------
        # Cargar entorno
        # ---------------------
        self.grid = np.array(env_cfg["grid"])
        self.start = tuple(env_cfg["start"])
        self.goal = tuple(env_cfg["goal"])

        self.height, self.width = self.grid.shape

        # Dimensiones (núcleo)
        self.state_dim = 6
        self.action_space_n = 4

        # 👉 Compatibilidad tipo Gym (CLAVE)
        self.observation_space = self.state_dim
        self.action_space = self.action_space_n

        self.max_steps = env_cfg.get("max_steps", 500)

        self.agent_pos = None
        self.steps = 0

        # Factory (clave para evaluación y smoke tests)
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
                float(self._is_wall(ax - 1, ay)),
                float(self._is_wall(ax + 1, ay)),
            ],
            dtype=np.float32,
        )

    def _is_wall(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return self.grid[x, y] == 1

