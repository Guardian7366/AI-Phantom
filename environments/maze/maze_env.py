from typing import Callable
import numpy as np
import random


class MazeEnvironment:

    ACTIONS = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    def __init__(self, config: dict):
        self.config = config

        if "grid" not in config:
            raise KeyError("MazeEnvironment requiere 'grid' base")

        self.base_grid = np.array(config["grid"], dtype=np.int32)
        self.height, self.width = self.base_grid.shape

        self.max_steps = config.get("max_steps", 500)

        # Curriculum level
        self.curriculum_level = 0

        self.channels = 4
        self.state_shape = (self.channels, self.height, self.width)
        self.state_dim = self.state_shape

        self.action_space_n = 4
        self.action_space = self.action_space_n
        self.observation_space = self.state_shape

        self.factory: Callable[[], "MazeEnvironment"] = (
            lambda: MazeEnvironment(self.config)
        )

    # ======================================================
    # CURRICULUM CONTROL
    # ======================================================

    def set_curriculum_level(self, level: int):
        self.curriculum_level = level

    def _apply_curriculum(self):
        if self.curriculum_level == 0:
            self.randomize_grid = False
            self.random_start_goal = False
            self.wall_probability = 0.0

        elif self.curriculum_level == 1:
            self.randomize_grid = False
            self.random_start_goal = True
            self.wall_probability = 0.05

        elif self.curriculum_level == 2:
            self.randomize_grid = True
            self.random_start_goal = True
            self.wall_probability = 0.08

        else:
            self.randomize_grid = True
            self.random_start_goal = True
            self.wall_probability = 0.12

    # ======================================================

    def _generate_random_grid(self):
        grid = np.zeros((self.height, self.width), dtype=np.int32)
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

    # ======================================================

    def reset(self):
        self._apply_curriculum()
        self.steps = 0

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

        self.visit_counts = np.zeros((self.height, self.width), dtype=np.float32)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1

        return self._get_state()

    # ======================================================

    def step(self, action: int):
        self.steps += 1

        dx, dy = self.ACTIONS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = -0.015
        done = False
        info = {"success": False}

        old_dist = self._manhattan_distance(self.agent_pos, self.goal)

        if self._is_wall(nx, ny):
            reward -= 0.1
        else:
            self.agent_pos = [nx, ny]

            visit_count = self.visit_counts[nx, ny]
            if visit_count > 0:
                reward -= 0.01 * min(visit_count, 5)

            self.visit_counts[nx, ny] += 1

        new_dist = self._manhattan_distance(self.agent_pos, self.goal)

        max_dist = self.height + self.width

        old_potential = -old_dist / max_dist
        new_potential = -new_dist / max_dist
        reward += new_potential - old_potential # Modelo 2.6.2

        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            done = True
            info["success"] = True

        if self.steps >= self.max_steps:
            reward -= 0.25 # Modelo 2.5.3
            done = True

        return self._get_state(), reward, done, info

    # ======================================================

    def _get_state(self):
        walls = self.grid.astype(np.float32)

        agent_layer = np.zeros_like(walls)
        agent_layer[self.agent_pos[0], self.agent_pos[1]] = 1.0

        goal_layer = np.zeros_like(walls)
        goal_layer[self.goal[0], self.goal[1]] = 1.0

        visits = self.visit_counts.copy()
        if visits.max() > 0:
            visits = visits / visits.max()

        return np.stack([walls, agent_layer, goal_layer, visits], axis=0).astype(np.float32)

    def _is_wall(self, x, y):
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return self.grid[x, y] == 1

    def _manhattan_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
