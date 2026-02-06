from typing import Dict, Tuple, Any
import numpy as np

class MazeEnv:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal

        self.height, self.width = grid.shape
        self.visited = np.zeros_like(grid, dtype=int)

        self.agent_pos = None
        self.done = False

    def reset(self) -> Dict[str, Any]:
        self.agent_pos = list(self.start)
        self.visited.fill(0)
        self.done = False

        return self._emit_raw_state(
            collided=False,
            moved=False,
            terminal_type=None
        )

    def step(self, action: int) -> Dict[str, Any]:
        if self.done:
            raise RuntimeError("Episode has ended. Call reset().")

        dx, dy = self._action_to_delta(action)
        next_x = self.agent_pos[0] + dx
        next_y = self.agent_pos[1] + dy

        collided = False
        moved = False
        terminal_type = None

        if self._is_wall(next_x, next_y):
            collided = True
        else:
            self.agent_pos = [next_x, next_y]
            moved = True

        self.visited[self.agent_pos[0], self.agent_pos[1]] += 1

        if tuple(self.agent_pos) == self.goal:
            self.done = True
            terminal_type = "success"

        return self._emit_raw_state(
            collided=collided,
            moved=moved,
            terminal_type=terminal_type
        )

    # ------------------------
    # Raw facts only
    # ------------------------

    def _emit_raw_state(
        self,
        collided: bool,
        moved: bool,
        terminal_type: str | None
    ) -> Dict[str, Any]:

        x, y = self.agent_pos
        cell_value = self.grid[x, y]

        return {
            "agent": {
                "x": x,
                "y": y,
                "orientation": None
            },
            "cell": {
                "type": self._cell_type(cell_value),
                "properties": {}
            },
            "maze": {
                "width": self.width,
                "height": self.height
            },
            "events": {
                "collided": collided,
                "moved": moved
            },
            "terminal": {
                "done": self.done,
                "terminal_type": terminal_type
            },
            "memory": {
                "visited_count": int(self.visited[x, y])
            }
        }

    # ------------------------
    # Helpers (still env logic)
    # ------------------------

    def _is_wall(self, x: int, y: int) -> bool:
        if x < 0 or y < 0 or x >= self.height or y >= self.width:
            return True
        return self.grid[x, y] == 1

    def _cell_type(self, value: int) -> str:
        if value == 0:
            return "empty"
        if value == 1:
            return "wall"
        if value == 2:
            return "goal"
        return "unknown"

    def _action_to_delta(self, action: int) -> Tuple[int, int]:
        # 0: up, 1: right, 2: down, 3: left
        return {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)
        }.get(action, (0, 0))
