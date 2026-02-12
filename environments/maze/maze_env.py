from typing import Callable, Optional, List, Tuple
import numpy as np
import random


class MazeEnvironment:
    """
    Maze Environment v2 - Hardened

    Nuevas capacidades:
    - Random start/goal opcional
    - Reward shaping por distancia
    - Sensores 4-direccionales
    - Tamaño variable
    - Total backward compatibility
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    def __init__(self, config: dict):
        self.config = config

        if "environment" in config:
            env_cfg = config["environment"]
        else:
            env_cfg = config

        required_keys = ["grid"]
        for key in required_keys:
            if key not in env_cfg:
                raise KeyError(f"MazeEnvironment: falta clave '{key}'")

        self._load_from_config(env_cfg)

        # Sensores ampliados
        self.state_dim = 8
        self.action_space_n = 4

        self.observation_space = self.state_dim
        self.action_space = self.action_space_n

        self.agent_pos = None
        self.steps = 0

        self.factory: Callable[[], "MazeEnvironment"] = (
            lambda: MazeEnvironment(self.config)
        )

    # -------------------------------------------------

    def _load_from_config(self, env_cfg: dict):
        self.grid = np.array(env_cfg["grid"])
        self.height, self.width = self.grid.shape

        self.max_steps = env_cfg.get("max_steps", 500)

        self.random_start_goal = env_cfg.get("random_start_goal", False)

        self.start = tuple(env_cfg.get("start", (0, 0)))
        self.goal = tuple(env_cfg.get("goal", (self.height - 1, self.width - 1)))

    # -------------------------------------------------

    def _sample_free_cell(self):
        while True:
            x = random.randint(0, self.height - 1)
            y = random.randint(0, self.width - 1)
            if self.grid[x, y] == 0:
                return (x, y)

    # -------------------------------------------------

    def reset(self) -> np.ndarray:
        self.prev_action = None
        if self.random_start_goal:
            self.start = self._sample_free_cell()
            self.goal = self._sample_free_cell()
            while self.goal == self.start:
                self.goal = self._sample_free_cell()

        self.agent_pos = list(self.start)
        self.steps = 0
        return self._get_state()

    # -------------------------------------------------

    def step(self, action: int):
        reverse_actions = {
                0: 2,
                2: 0,
                1: 3,
                3: 1,
                }
        
        self.steps += 1

        dx, dy = self.ACTIONS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = -0.01
        done = False
        info = {"success": False}

        old_dist = self._manhattan_distance(self.agent_pos, self.goal)

        if self._is_wall(nx, ny):
            reward -= 0.1
            # Penalización por reversa inmediata (anti-loop)
            if self.prev_action is not None:
                if action == reverse_actions[self.prev_action]:
                    reward -= 0.05
        else:
            self.agent_pos = [nx, ny]

        new_dist = self._manhattan_distance(self.agent_pos, self.goal)

        # -------------------------------------------------
        # Potential-based reward shaping NORMALIZADO
        # -------------------------------------------------

        gamma = 0.99

        max_possible_dist = self.height + self.width

        old_potential = -old_dist / max_possible_dist
        new_potential = -new_dist / max_possible_dist

        shaping = gamma * new_potential - old_potential

        reward += shaping

        # -------------------------------------------------

        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True
            info["success"] = True

        if self.steps >= self.max_steps:
            done = True
            
        self.prev_action = action
        return self._get_state(), reward, done, info




    # -------------------------------------------------

    def _get_state(self) -> np.ndarray:
        ax, ay = self.agent_pos
        gx, gy = self.goal

        return np.array(
            [
                ax / self.height,
                ay / self.width,
                (gx - ax) / self.height,
                (gy - ay) / self.width,

                # Sensores completos
                float(self._is_wall(ax - 1, ay)),  # up
                float(self._is_wall(ax + 1, ay)),  # down
                float(self._is_wall(ax, ay - 1)),  # left
                float(self._is_wall(ax, ay + 1)),  # right
            ],
            dtype=np.float32,
        )


    # -------------------------------------------------

    def _is_wall(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return self.grid[x, y] == 1

    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
