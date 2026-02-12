from typing import Callable
import numpy as np
import random


class MazeEnvironment:
    """
    Maze Environment v4.0
    - Memory augmented state
    - Anti-loop revisitation penalty
    - Potential based reward shaping
    - Backward compatible API
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }

    # =====================================================
    # INIT
    # =====================================================

    def __init__(self, config: dict):
        self.config = config

        if "environment" in config:
            env_cfg = config["environment"]
        else:
            env_cfg = config

        if "grid" not in env_cfg:
            raise KeyError("MazeEnvironment requiere 'grid' base")

        self.base_grid = np.array(env_cfg["grid"])
        self.height, self.width = self.base_grid.shape

        self.random_start_goal = env_cfg.get("random_start_goal", False)
        self.randomize_grid = env_cfg.get("randomize_grid", False)
        self.wall_probability = env_cfg.get("wall_probability", 0.25)
        self.max_steps = env_cfg.get("max_steps", 500)

        # =====================================================
        # STATE DIMENSION (CRÍTICO)
        # 8 base + 8 prev_state + 4 prev_action = 20
        # =====================================================
        self.base_state_dim = 8
        self.state_dim = 20
        self.action_space_n = 4

        self.observation_space = self.state_dim
        self.action_space = self.action_space_n

        # Runtime variables
        self.agent_pos = None
        self.goal = None
        self.prev_state = None
        self.prev_action = None
        self.steps = 0

        self.factory: Callable[[], "MazeEnvironment"] = (
            lambda: MazeEnvironment(self.config)
        )

    # =====================================================
    # GRID GENERATION
    # =====================================================

    def _generate_random_grid(self):
        grid = np.zeros((self.height, self.width), dtype=int)
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < self.wall_probability:
                    grid[i, j] = 1
        return grid

    def _sample_free_cell(self):
        while True:
            x = random.randint(0, self.height - 1)
            y = random.randint(0, self.width - 1)
            if self.grid[x, y] == 0:
                return (x, y)

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):
        self.steps = 0
        self.prev_state = None
        self.prev_action = None

        if self.randomize_grid:
            self.grid = self._generate_random_grid()
        else:
            self.grid = self.base_grid.copy()

        if self.random_start_goal:
            self.start = self._sample_free_cell()
            self.goal = self._sample_free_cell()
            while self.goal == self.start:
                self.goal = self._sample_free_cell()
        else:
            self.start = (0, 0)
            self.goal = (self.height - 1, self.width - 1)

        self.agent_pos = list(self.start)

        self.visit_counts = np.zeros((self.height, self.width), dtype=np.int32)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1

        return self._get_state()

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action: int):
        reverse_actions = {0: 2, 2: 0, 1: 3, 3: 1}
        self.steps += 1

        old_state_base = self._get_base_state()

        dx, dy = self.ACTIONS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = -0.01
        done = False
        info = {"success": False}

        old_dist = self._manhattan_distance(self.agent_pos, self.goal)

        if self._is_wall(nx, ny):
            reward -= 0.1
            if self.prev_action is not None:
                if action == reverse_actions[self.prev_action]:
                    reward -= 0.05
        else:
            self.agent_pos = [nx, ny]

            visit_count = self.visit_counts[nx, ny]
            if visit_count > 0:
                reward -= 0.02 * visit_count

            self.visit_counts[nx, ny] += 1

        new_dist = self._manhattan_distance(self.agent_pos, self.goal)

        gamma = 0.99
        max_possible_dist = self.height + self.width

        old_potential = -old_dist / max_possible_dist
        new_potential = -new_dist / max_possible_dist
        reward += gamma * new_potential - old_potential

        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True
            info["success"] = True

        if self.steps >= self.max_steps:
            done = True

        # Guardar memoria
        self.prev_state = old_state_base
        self.prev_action = action

        return self._get_state(), reward, done, info

    # =====================================================
    # STATE CONSTRUCTION
    # =====================================================

    def _get_base_state(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal

        # Normalización espacial
        ax /= self.height
        ay /= self.width
        gx /= self.height
        gy /= self.width

        wall_up = int(self._is_wall(self.agent_pos[0] - 1, self.agent_pos[1]))
        wall_right = int(self._is_wall(self.agent_pos[0], self.agent_pos[1] + 1))
        wall_down = int(self._is_wall(self.agent_pos[0] + 1, self.agent_pos[1]))
        wall_left = int(self._is_wall(self.agent_pos[0], self.agent_pos[1] - 1))

        return np.array([
            ax, ay,
            gx, gy,
            wall_up,
            wall_right,
            wall_down,
            wall_left
        ], dtype=np.float32)

    def _get_state(self):
        current_state = self._get_base_state()

        if self.prev_state is None:
            prev_state = np.zeros(self.base_state_dim, dtype=np.float32)
        else:
            prev_state = self.prev_state

        if self.prev_action is None:
            prev_action = np.zeros(self.action_space, dtype=np.float32)
        else:
            prev_action = np.eye(self.action_space, dtype=np.float32)[self.prev_action]

        augmented_state = np.concatenate([
            current_state,
            prev_state,
            prev_action
        ])

        return augmented_state

    # =====================================================
    # HELPERS
    # =====================================================

    def _is_wall(self, x: int, y: int):
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return self.grid[x, y] == 1

    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
