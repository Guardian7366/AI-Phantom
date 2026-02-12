from typing import Callable, Optional, List, Tuple
import numpy as np


class MazeEnvironment:
    """
    Entorno de laberinto compatible con DQN.
    Soporta:
    - Config estático (modo actual)
    - Actualización dinámica de grid (Sandbox ready)
    """

    ACTIONS = {
        0: (-1, 0),  # up (row - 1)
        1: (0, 1),   # right (col + 1)
        2: (1, 0),   # down (row + 1)
        3: (0, -1),  # left (col - 1)
    }

    def __init__(self, config: dict):
        self.config = config

        if "environment" in config:
            env_cfg = config["environment"]
        else:
            env_cfg = config

        required_keys = ["grid", "start", "goal"]
        for key in required_keys:
            if key not in env_cfg:
                raise KeyError(
                    f"MazeEnvironment: falta clave '{key}' en config"
                )

        self._load_from_config(env_cfg)

        # Dimensiones núcleo
        self.state_dim = 6
        self.action_space_n = 4
        self.observation_space = self.state_dim
        self.action_space = self.action_space_n

        self.agent_pos = None
        self.steps = 0

        # Factory para evaluación / smoke test
        self.factory: Callable[[], "MazeEnvironment"] = (
            lambda: MazeEnvironment(self.config)
        )

    # -------------------------------------------------
    # Carga interna
    # -------------------------------------------------

    def _load_from_config(self, env_cfg: dict):
        self.grid = np.array(env_cfg["grid"])
        self.start = tuple(env_cfg["start"])
        self.goal = tuple(env_cfg["goal"])
        self.height, self.width = self.grid.shape
        self.max_steps = env_cfg.get("max_steps", 500)

    # -------------------------------------------------
    # NUEVO: actualización dinámica (Sandbox)
    # -------------------------------------------------

    def set_grid(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_steps: Optional[int] = None,
    ):
        """
        Permite cambiar el laberinto dinámicamente
        sin reinstanciar el entorno.
        """
        self.grid = np.array(grid)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.height, self.width = self.grid.shape

        if max_steps is not None:
            self.max_steps = max_steps

        # Reset interno seguro
        self.agent_pos = None
        self.steps = 0

    # -------------------------------------------------
    # Core API
    # -------------------------------------------------

    def reset(self) -> np.ndarray:
        # 🔑 Generar nuevo maze si es procedural
        if hasattr(self, "gen_cfg"):
            self._generate_maze()

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

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _get_state(self) -> np.ndarray:
        ax, ay = self.agent_pos
        gx, gy = self.goal

        return np.array([
            ax / self.height,
            ay / self.width,
            (gx - ax) / self.height,
            (gy - ay) / self.width,
            float(self._is_wall(ax - 1, ay)),
            float(self._is_wall(ax + 1, ay)),
        ], dtype=np.float32)

    def _is_wall(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return self.grid[x, y] == 1
