from typing import Callable
import numpy as np

from .maze_generator import generate_dfs_maze


class MazeEnvironment:
    """
    Entorno de laberinto compatible con DQN.

    Ahora soporta:
    - Maze predefinido (grid en YAML)
    - Maze procedural (DFS generator + seed)
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

        # -------------------------------------------------
        # Procedural Maze Mode
        # -------------------------------------------------
        if "generator" in env_cfg:
            self.gen_cfg = env_cfg["generator"]

            self.width = self.gen_cfg.get("width", 7)
            self.height = self.gen_cfg.get("height", 7)
            self.start = tuple(self.gen_cfg.get("start", (1, 1)))
            self.goal = tuple(self.gen_cfg.get("goal", (self.height - 2, self.width - 2)))

            self.loop_prob = self.gen_cfg.get("loop_probability", 0.05)
            self.seed = self.gen_cfg.get("seed", None)

            # Generar primer maze
            self._generate_maze()

        # -------------------------------------------------
        # Static Maze Mode (Backward Compatibility)
        # -------------------------------------------------
        else:
            required_keys = ["grid", "start", "goal"]
            for key in required_keys:
                if key not in env_cfg:
                    raise KeyError(f"MazeEnvironment: falta clave '{key}' en config")

            self.grid = np.array(env_cfg["grid"])
            self.start = tuple(env_cfg["start"])
            self.goal = tuple(env_cfg["goal"])

            self.height, self.width = self.grid.shape

        # -------------------------------------------------
        self.state_dim = 6
        self.action_space_n = 4
        self.observation_space = self.state_dim
        self.action_space = self.action_space_n

        self.max_steps = env_cfg.get("max_steps", 500)
        self.agent_pos = None
        self.steps = 0

        # Factory para evaluaciones / entrenamiento
        self.factory: Callable[[], "MazeEnvironment"] = lambda: MazeEnvironment(self.config)

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

    def _generate_maze(self):
        """Genera un nuevo laberinto procedural."""
        self.grid = generate_dfs_maze(
            width=self.width,
            height=self.height,
            seed=self.seed,  # puede ser None para aleatorio
            loop_probability=self.loop_prob
        )
        self.height, self.width = self.grid.shape

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
