import numpy as np
from typing import Tuple, Dict


class MazeEnvironment:
    """
    Laberinto 2D discreto con movimiento en 4 direcciones.
    Entorno totalmente desacoplado de cualquier agente o algoritmo.
    """

    ACTIONS = {
        0: (0, -1),   # up
        1: (0, 1),    # down
        2: (-1, 0),   # left
        3: (1, 0),    # right
    }

    def __init__(
        self,
        maze: np.ndarray,
        max_steps: int = 200,
    ):
        """
        maze: matriz 2D (0 libre, 1 pared)
        max_steps: límite duro por episodio
        """
        self.maze = maze
        self.height, self.width = maze.shape
        self.max_steps = max_steps

        self.agent_pos = None
        self.goal_pos = None
        self.steps = 0
        self.last_action = None
        self.visited_counter = {}

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.agent_pos = self._random_free_position()
        self.goal_pos = self._random_free_position(exclude=self.agent_pos)

        self.steps = 0
        self.last_action = None
        self.visited_counter = {self.agent_pos: 1}

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.steps += 1

        collided = False
        new_pos = self._attempt_move(action)

        if new_pos == self.agent_pos:
            collided = True
        else:
            self.agent_pos = new_pos

        self.last_action = action
        self.visited_counter[self.agent_pos] = self.visited_counter.get(self.agent_pos, 0) + 1

        reward = self._compute_reward(collided)
        done, terminal_type = self._check_terminal()

        info = {
            "collided": collided,
            "distance_to_goal": self._manhattan_distance(),
            "terminal_type": terminal_type,
        }

        return self._get_state(), reward, done, info

    def get_action_space(self) -> int:
        return len(self.ACTIONS)

    def get_state_space(self) -> int:
        return len(self._get_state())

    # ------------------------------------------------------------------
    # INTERNAL LOGIC
    # ------------------------------------------------------------------

    def _attempt_move(self, action: int) -> Tuple[int, int]:
        dx, dy = self.ACTIONS[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        if self._is_wall(nx, ny):
            return self.agent_pos

        return (nx, ny)

    def _is_wall(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return True
        return self.maze[y, x] == 1

    def _manhattan_distance(self) -> int:
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        return abs(ax - gx) + abs(ay - gy)

    def _compute_reward(self, collided: bool) -> float:
        reward = -0.01  # paso de tiempo

        if collided:
            reward -= 0.1

        distance = self._manhattan_distance()
        reward += -0.01 * distance

        if self.agent_pos == self.goal_pos:
            reward += 1.0

        if self.visited_counter[self.agent_pos] > 3:
            reward -= 0.05

        return reward

    def _check_terminal(self) -> Tuple[bool, str]:
        if self.agent_pos == self.goal_pos:
            return True, "success"

        if self.steps >= self.max_steps:
            return True, "timeout"

        if self.visited_counter[self.agent_pos] > 10:
            return True, "stuck"

        return False, "none"

    def _get_state(self) -> np.ndarray:
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos

        dx = (gx - ax) / self.width
        dy = (gy - ay) / self.height

        distance = self._manhattan_distance() / (self.width + self.height)

        walls = [
            int(self._is_wall(ax, ay - 1)),  # up
            int(self._is_wall(ax, ay + 1)),  # down
            int(self._is_wall(ax - 1, ay)),  # left
            int(self._is_wall(ax + 1, ay)),  # right
        ]

        last_move = [0, 0, 0, 0]
        if self.last_action is not None:
            last_move[self.last_action] = 1

        stuck_flag = int(self.visited_counter.get(self.agent_pos, 0) > 1)

        state = np.array(
            [dx, dy, distance] + walls + last_move + [stuck_flag],
            dtype=np.float32
        )

        return state

    def _random_free_position(self, exclude=None) -> Tuple[int, int]:
        while True:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.maze[y, x] == 0:
                pos = (x, y)
                if exclude is None or pos != exclude:
                    return pos
