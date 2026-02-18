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

        self.randomize_grid = config.get("randomize_grid", False)
        self.random_start_goal = config.get("random_start_goal", False)
        self.wall_probability = config.get("wall_probability", 0.0)

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

        elif self.curriculum_level == 3:
            self.randomize_grid = True
            self.random_start_goal = True
            self.wall_probability = 0.12

        elif self.curriculum_level == 4:
            self.randomize_grid = True
            self.random_start_goal = True
            self.wall_probability = 0.15

        else:
            self.randomize_grid = True
            self.random_start_goal = True
            self.wall_probability = 0.18



    # ======================================================

    def _generate_random_grid(self):
        grid = np.zeros((self.height, self.width), dtype=np.int32)
        for i in range(self.height):
            for j in range(self.width):
                if random.random() < self.wall_probability:
                    grid[i, j] = 1
        return grid
    
    # ======================================================
    # Connectivity Check (Modelo 2.7.2)
    # ======================================================

    def _is_reachable(self, grid, start, goal):
        from collections import deque

        if grid[start[0], start[1]] == 1:
            return False
        if grid[goal[0], goal[1]] == 1:
            return False

        visited = np.zeros_like(grid, dtype=bool)
        queue = deque([start])
        visited[start[0], start[1]] = True

        while queue:
            x, y = queue.popleft()

            if (x, y) == goal:
                return True

            for dx, dy in self.ACTIONS.values():
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < self.height
                    and 0 <= ny < self.width
                    and not visited[nx, ny]
                    and grid[nx, ny] == 0
                ):
                    visited[nx, ny] = True
                    queue.append((nx, ny))

        return False


    def _sample_free_cell(self):
        while True:
            x = random.randint(0, self.height - 1)
            y = random.randint(0, self.width - 1)
            if self.grid[x, y] == 0:
                return (x, y)

    def reset(self):
        self._apply_curriculum()
        self.steps = 0

        max_attempts = 50

        for _ in range(max_attempts):

            # --------------------------------------------------
            # Generar grid candidato
            # --------------------------------------------------
            if self.randomize_grid:
                candidate_grid = self._generate_random_grid()

                # Calcular densidad de muros
                wall_ratio = np.mean(candidate_grid)

                # Evitar mapas demasiado cerrados
                if wall_ratio > 0.35:
                    continue
            else:
                candidate_grid = self.base_grid.copy()

            # --------------------------------------------------
            # Sampling start / goal
            # --------------------------------------------------
            if self.random_start_goal:
                free_cells = np.argwhere(candidate_grid == 0)

                if len(free_cells) < 2:
                    continue

                start_idx = random.randint(0, len(free_cells) - 1)
                goal_idx = random.randint(0, len(free_cells) - 1)

                start = tuple(free_cells[start_idx])
                goal = tuple(free_cells[goal_idx])

                if start == goal:
                    continue

                min_distance = (self.height + self.width) // 3
                if self._manhattan_distance(start, goal) < min_distance:
                    continue
            else:
                start = (0, 0)
                goal = (self.height - 1, self.width - 1)

            # --------------------------------------------------
            # Verificar conectividad
            # --------------------------------------------------
            if self._is_reachable(candidate_grid, start, goal):
                self.grid = candidate_grid
                self.start = start
                self.goal = goal
                break
        else:
            # Fallback seguro
            self.grid = self.base_grid.copy()
            self.start = (0, 0)
            self.goal = (self.height - 1, self.width - 1)

        # --------------------------------------------------
        # Inicializar estado
        # --------------------------------------------------
        self.agent_pos = list(self.start)

        self.visit_counts = np.zeros(
            (self.height, self.width),
            dtype=np.float32
        )
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1

        return self._get_state()



    def step(self, action: int):
        self.steps += 1

        dx, dy = self.ACTIONS[action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy

        reward = -0.01  # small living cost
        done = False
        info = {"success": False}

        old_dist = self._manhattan_distance(self.agent_pos, self.goal)

        # --------------------------------------------------
        # Movimiento
        # --------------------------------------------------
        if self._is_wall(nx, ny):
            reward -= 0.15  # reducido (antes 0.3)
        else:
            self.agent_pos = [nx, ny]

            # Bonus por celda nueva
            if self.visit_counts[nx, ny] == 0:
                reward += 0.05
            else:
                reward -= 0.01

            self.visit_counts[nx, ny] += 1

        new_dist = self._manhattan_distance(self.agent_pos, self.goal)

        # --------------------------------------------------
        # Distance shaping suavizado
        # --------------------------------------------------
        if new_dist < old_dist:
            reward += 0.15   # antes 0.3
        elif new_dist > old_dist:
            reward -= 0.05   # antes 0.1

        # --------------------------------------------------
        # Goal
        # --------------------------------------------------
        if tuple(self.agent_pos) == self.goal:
            reward = 20.0   # ligeramente más alto
            done = True
            info["success"] = True

        # --------------------------------------------------
        # Max steps
        # --------------------------------------------------
        if self.steps >= self.max_steps:
            reward -= 0.5
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
