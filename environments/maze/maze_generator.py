# environments/maze/maze_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any, Iterable
import heapq
import numpy as np

# Import seguro (maze_env NO importa maze_generator, así que no hay ciclo)
from environments.maze.maze_env import ACTIONS, MazeEnvironment


Pos = Tuple[int, int]


def _invert_actions(actions: Dict[int, Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    inv: Dict[Tuple[int, int], int] = {}
    for a, (dr, dc) in actions.items():
        inv[(int(dr), int(dc))] = int(a)
    return inv


_INV_ACTIONS = _invert_actions(ACTIONS)


@dataclass(frozen=True)
class ExpertConfig:
    algorithm: str = "bfs"   # "bfs" | "astar"
    max_nodes: int = 200000  # límite defensivo


class MazeExpertPlanner:
    """
    Experto clásico para gridworld:
    - BFS: shortest path (coste uniforme)
    - A*: shortest path (heurística Manhattan)
    Devuelve path de posiciones y acciones {0,1,2,3} compatibles con MazeEnvironment.
    """

    def __init__(self, cfg: Optional[ExpertConfig] = None):
        self.cfg = cfg or ExpertConfig()

    # -------------------------
    # API principal
    # -------------------------
    def plan(
        self,
        grid: np.ndarray,
        start: Pos,
        goal: Pos,
    ) -> Optional[List[int]]:
        """
        Retorna lista de acciones (0..3) que llevan de start a goal.
        Retorna None si no hay path.
        """
        algo = str(self.cfg.algorithm).lower().strip()
        if algo == "astar":
            path = self._astar_path(grid, start, goal)
        else:
            path = self._bfs_path(grid, start, goal)

        if path is None or len(path) < 2:
            return None

        actions = self.path_to_actions(path)
        return actions

    def plan_from_env(self, env: MazeEnvironment) -> Optional[List[int]]:
        return self.plan(env.grid, env.agent_pos, env.goal_pos)

    # -------------------------
    # Converters
    # -------------------------
    @staticmethod
    def path_to_actions(path: List[Pos]) -> List[int]:
        """
        Convierte [pos0,pos1,...] a acciones (0..3).
        """
        out: List[int] = []
        for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
            dr, dc = int(r1 - r0), int(c1 - c0)
            a = _INV_ACTIONS.get((dr, dc), None)
            if a is None:
                # path inválido (no adyacente) -> abort
                return []
            out.append(int(a))
        return out

    # -------------------------
    # BFS shortest path
    # -------------------------
    def _bfs_path(self, grid: np.ndarray, start: Pos, goal: Pos) -> Optional[List[Pos]]:
        H, W = int(grid.shape[0]), int(grid.shape[1])
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])

        if not self._in_bounds(sr, sc, H, W) or not self._in_bounds(gr, gc, H, W):
            return None
        if int(grid[sr, sc]) == 1 or int(grid[gr, gc]) == 1:
            return None
        if (sr, sc) == (gr, gc):
            return [(sr, sc)]

        # parent pointers: -1 significa "no visitado"
        parent_r = np.full((H, W), -1, dtype=np.int16)
        parent_c = np.full((H, W), -1, dtype=np.int16)

        q: List[Pos] = [(sr, sc)]
        head = 0
        parent_r[sr, sc] = sr
        parent_c[sr, sc] = sc

        nodes = 0
        while head < len(q):
            r, c = q[head]
            head += 1
            nodes += 1
            if nodes > int(self.cfg.max_nodes):
                return None

            for dr, dc in ACTIONS.values():
                nr, nc = r + int(dr), c + int(dc)
                if not self._in_bounds(nr, nc, H, W):
                    continue
                if int(grid[nr, nc]) == 1:
                    continue
                if parent_r[nr, nc] != -1:
                    continue

                parent_r[nr, nc] = r
                parent_c[nr, nc] = c

                if (nr, nc) == (gr, gc):
                    return self._reconstruct_path(parent_r, parent_c, (sr, sc), (gr, gc))

                q.append((nr, nc))

        return None

    # -------------------------
    # A* shortest path
    # -------------------------
    def _astar_path(self, grid: np.ndarray, start: Pos, goal: Pos) -> Optional[List[Pos]]:
        H, W = int(grid.shape[0]), int(grid.shape[1])
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])

        if not self._in_bounds(sr, sc, H, W) or not self._in_bounds(gr, gc, H, W):
            return None
        if int(grid[sr, sc]) == 1 or int(grid[gr, gc]) == 1:
            return None
        if (sr, sc) == (gr, gc):
            return [(sr, sc)]

        # g score (cost from start)
        gscore = np.full((H, W), np.iinfo(np.int32).max, dtype=np.int32)
        gscore[sr, sc] = 0

        # parent pointers
        parent_r = np.full((H, W), -1, dtype=np.int16)
        parent_c = np.full((H, W), -1, dtype=np.int16)
        parent_r[sr, sc] = sr
        parent_c[sr, sc] = sc

        # open set heap: (f, g, r, c)
        h0 = self._manhattan(sr, sc, gr, gc)
        heap: List[Tuple[int, int, int, int]] = [(h0, 0, sr, sc)]

        nodes = 0
        while heap:
            f, g, r, c = heapq.heappop(heap)
            nodes += 1
            if nodes > int(self.cfg.max_nodes):
                return None

            if (r, c) == (gr, gc):
                return self._reconstruct_path(parent_r, parent_c, (sr, sc), (gr, gc))

            # stale pop
            if g != int(gscore[r, c]):
                continue

            for dr, dc in ACTIONS.values():
                nr, nc = r + int(dr), c + int(dc)
                if not self._in_bounds(nr, nc, H, W):
                    continue
                if int(grid[nr, nc]) == 1:
                    continue

                ng = g + 1
                if ng < int(gscore[nr, nc]):
                    gscore[nr, nc] = ng
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c
                    nf = ng + self._manhattan(nr, nc, gr, gc)
                    heapq.heappush(heap, (int(nf), int(ng), int(nr), int(nc)))

        return None

    # -------------------------
    # Utils
    # -------------------------
    @staticmethod
    def _in_bounds(r: int, c: int, H: int, W: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    @staticmethod
    def _manhattan(r: int, c: int, gr: int, gc: int) -> int:
        return abs(int(r) - int(gr)) + abs(int(c) - int(gc))

    @staticmethod
    def _reconstruct_path(
        parent_r: np.ndarray,
        parent_c: np.ndarray,
        start: Pos,
        goal: Pos,
    ) -> List[Pos]:
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])

        path: List[Pos] = [(gr, gc)]
        r, c = gr, gc
        # parent pointers garantizan que start apunta a sí mismo
        while not (r == sr and c == sc):
            pr = int(parent_r[r, c])
            pc = int(parent_c[r, c])
            # defensivo: si algo sale mal
            if pr < 0 or pc < 0:
                break
            r, c = pr, pc
            path.append((r, c))
        path.reverse()
        return path


# ---------------------------------------------------------------------
# Helpers opcionales para recolección (sin acoplarlo al trainer)
# ---------------------------------------------------------------------
def rollout_expert_episode(
    env: MazeEnvironment,
    planner: MazeExpertPlanner,
    *,
    curriculum_level: int,
    seed: Optional[int] = None,
    freeze_pool: bool = True,
    max_steps_cap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ejecuta 1 episodio siguiendo el experto y regresa un dict con:
    - obs_list: lista de obs (np.ndarray (3,H,W))
    - act_list: lista de acciones (int)
    - info0: info de reset (incluye bfs_dist_start, family, etc.)
    - reached_goal: bool
    Nota: útil para "dataset builder" luego, sin tocar DQN todavía.
    """
    obs, info0 = env.reset(
        curriculum_level=int(curriculum_level),
        seed=seed,
        freeze_pool=bool(freeze_pool),
    )

    actions = planner.plan_from_env(env)
    if actions is None:
        return {
            "obs_list": [],
            "act_list": [],
            "info0": info0,
            "reached_goal": False,
            "failed_reason": "no_path",
        }

    max_cap = int(max_steps_cap) if max_steps_cap is not None else int(getattr(env.cfg, "max_steps", 128))
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []

    reached = False
    for t, a in enumerate(actions):
        if t >= max_cap:
            break
        obs_list.append(obs)
        act_list.append(int(a))
        obs, r, terminated, truncated, info = env.step(int(a))
        if terminated:
            reached = True
            break
        if truncated:
            break

    return {
        "obs_list": obs_list,
        "act_list": act_list,
        "info0": info0,
        "reached_goal": bool(reached),
        "path_len": int(len(act_list)),
    }