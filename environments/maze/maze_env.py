# environments/maze/maze_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, Callable
from collections import deque

import numpy as np

ACTIONS = {
    0: (-1, 0),
    1: ( 1, 0),
    2: ( 0,-1),
    3: ( 0, 1),
}


@dataclass
class MazeConfig:
    size: int = 8
    wall_prob: float = 0.28
    max_gen_tries: int = 200
    max_steps: int = 128

    # Recompensas
    step_penalty: float = -0.01
    invalid_move_penalty: float = -0.02
    goal_reward: float = 1.0
    progress_reward: float = 0.02

    # Evita casos triviales / demasiado cortos
    min_bfs_start_goal_lvl1: int = 2
    min_bfs_start_goal_lvl2: int = 4

    # Pool de grids para lvl2 (estabilidad + generalización)
    lvl2_grid_pool_size: int = 64           # 0 => desactivado
    lvl2_pool_refresh_prob: float = 0.05    # ✅ default alineado con YAML "estable"


class MazeEnvironment:
    """
    Laberinto NxN con muros (1=wall, 0=free). Genera laberintos resolubles.
    Observación: (3, H, W) = (walls, agent, goal).

    Reproducibilidad / estabilidad:
    - reset(seed=...) reseedea el RNG del env para reproducibilidad por episodio.
    - En lvl2 con pool activo, el pool SOLO se resetea si reset_pool=True (y lvl>=2).
    - freeze_pool=True evita refreshes del pool durante evaluación/batch (señal comparable).
    """

    def __init__(self, config: Optional[MazeConfig] = None, rng_seed: int = 0):
        self.cfg = config or MazeConfig()
        self.rng = np.random.default_rng(int(rng_seed))

        self.size = int(self.cfg.size)
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (self.size - 1, self.size - 1)

        self.steps = 0
        self.last_bfs_dist: Optional[int] = None

        # Nivel 0/1: grid fijo
        self.fixed_grid_cache: Optional[np.ndarray] = None

        # Pool lvl2
        self._lvl2_pool: List[np.ndarray] = []
        self._lvl2_pool_size = max(0, int(getattr(self.cfg, "lvl2_grid_pool_size", 0)))
        self._lvl2_refresh_prob = float(getattr(self.cfg, "lvl2_pool_refresh_prob", 0.0))

        # BFS buffers reutilizables (performance)
        self._bfs_dist_grid = np.full((self.size, self.size), -1, dtype=np.int16)
        self._bfs_queue: deque = deque()

    # -------------------------
    # API
    # -------------------------
    def reset(
        self,
        *,
        curriculum_level: int = 0,
        seed: Optional[int] = None,
        reset_pool: bool = False,
        freeze_pool: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        lvl = int(curriculum_level)

        # Re-seed del RNG (reproducible por episodio)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        # ✅ Blindaje: solo lvl2 puede resetear pool (evita efectos colaterales)
        if bool(reset_pool) and lvl >= 2 and self._lvl2_pool_size > 0:
            self._lvl2_pool.clear()

        self.steps = 0

        if lvl <= 0:
            # Grid fijo, start/goal fijos
            if self.fixed_grid_cache is None:
                self.grid = self._generate_solvable_grid()
                self.fixed_grid_cache = self.grid.copy()
            else:
                self.grid = self.fixed_grid_cache.copy()

            self.agent_pos = (0, 0)
            self.goal_pos = (self.size - 1, self.size - 1)
            self.grid[self.agent_pos] = 0
            self.grid[self.goal_pos] = 0

        elif lvl == 1:
            # Grid fijo, start/goal aleatorios
            if self.fixed_grid_cache is None:
                self.grid = self._generate_solvable_grid()
                self.fixed_grid_cache = self.grid.copy()
            else:
                self.grid = self.fixed_grid_cache.copy()

            min_dist = int(getattr(self.cfg, "min_bfs_start_goal_lvl1", 0))
            self.agent_pos, self.goal_pos = self._sample_start_goal_robust(
                min_bfs=min_dist,
                max_grid_tries=1,      # grid fijo, no regen
                max_pair_tries=400,
                regen_grid_fn=None,    # ✅ nunca regenerar aquí
            )
            self.grid[self.agent_pos] = 0
            self.grid[self.goal_pos] = 0

        else:
            # lvl2: grid variable + start/goal aleatorios (pool si activo)
            min_dist = int(getattr(self.cfg, "min_bfs_start_goal_lvl2", 0))

            # ✅ Centraliza freeze_pool: si freeze_pool=True -> no refresh
            def _lvl2_grid() -> np.ndarray:
                return self._sample_or_build_lvl2_grid(
                    allow_refresh=(not bool(freeze_pool))
                )

            self.grid = _lvl2_grid()

            # Regen grid en lvl2 (si falla encontrar start/goal), respetando freeze_pool
            self.agent_pos, self.goal_pos = self._sample_start_goal_robust(
                min_bfs=min_dist,
                max_grid_tries=30,
                max_pair_tries=500,
                regen_grid_fn=_lvl2_grid,
            )
            self.grid[self.agent_pos] = 0
            self.grid[self.goal_pos] = 0

        self.last_bfs_dist = self._bfs_distance(self.agent_pos, self.goal_pos)

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "bfs_dist": self.last_bfs_dist,
            "curriculum_level": lvl,
            "freeze_pool": bool(freeze_pool),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1

        r = float(self.cfg.step_penalty)
        terminated = False
        truncated = False

        dr, dc = ACTIONS.get(int(action), (0, 0))
        ar, ac = self.agent_pos
        nr, nc = ar + dr, ac + dc

        invalid = (
            nr < 0 or nr >= self.size or
            nc < 0 or nc >= self.size or
            self.grid[nr, nc] == 1
        )

        if invalid:
            r += float(self.cfg.invalid_move_penalty)
            new_dist: Optional[int] = self.last_bfs_dist
        else:
            self.agent_pos = (nr, nc)
            new_dist = self._bfs_distance(self.agent_pos, self.goal_pos)

        if new_dist is None:
            new_dist = self.last_bfs_dist

        if self.last_bfs_dist is not None and new_dist is not None:
            delta = int(self.last_bfs_dist) - int(new_dist)
            r += float(self.cfg.progress_reward) * float(delta)

        self.last_bfs_dist = new_dist

        if self.agent_pos == self.goal_pos:
            r += float(self.cfg.goal_reward)
            terminated = True

        if self.steps >= int(self.cfg.max_steps) and not terminated:
            truncated = True

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "bfs_dist": self.last_bfs_dist,
            "steps": int(self.steps),
        }
        return obs, float(r), bool(terminated), bool(truncated), info

    # -------------------------
    # Obs
    # -------------------------
    def _get_obs(self) -> np.ndarray:
        walls = self.grid.astype(np.float32, copy=False)
        agent = np.zeros_like(walls, dtype=np.float32)
        goal = np.zeros_like(walls, dtype=np.float32)

        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        agent[ar, ac] = 1.0
        goal[gr, gc] = 1.0

        return np.stack([walls, agent, goal], axis=0)

    # -------------------------
    # Lvl2 pool helpers
    # -------------------------
    def _sample_or_build_lvl2_grid(self, force_rebuild: bool = False, allow_refresh: bool = True) -> np.ndarray:
        pool_size = int(self._lvl2_pool_size)
        if pool_size <= 0:
            return self._generate_solvable_grid()

        if bool(force_rebuild):
            self._lvl2_pool.clear()

        # Inicializa pool si vacío
        if len(self._lvl2_pool) == 0:
            for _ in range(pool_size):
                self._lvl2_pool.append(self._generate_solvable_grid())

        # Refresco opcional (diversidad controlada)
        if bool(allow_refresh) and (self.rng.random() < float(self._lvl2_refresh_prob)):
            i = int(self.rng.integers(0, len(self._lvl2_pool)))
            self._lvl2_pool[i] = self._generate_solvable_grid()

        # Samplea uno del pool
        j = int(self.rng.integers(0, len(self._lvl2_pool)))
        return self._lvl2_pool[j].copy()

    def _generate_solvable_grid(self) -> np.ndarray:
        p = float(self.cfg.wall_prob)
        p_edge = p * 0.6

        for _ in range(int(self.cfg.max_gen_tries)):
            grid = (self.rng.random((self.size, self.size)) < p).astype(np.int8)

            grid[0, :] = (self.rng.random(self.size) < p_edge).astype(np.int8)
            grid[-1, :] = (self.rng.random(self.size) < p_edge).astype(np.int8)
            grid[:, 0] = (self.rng.random(self.size) < p_edge).astype(np.int8)
            grid[:, -1] = (self.rng.random(self.size) < p_edge).astype(np.int8)

            grid[0, 0] = 0
            grid[self.size - 1, self.size - 1] = 0

            # solvable al menos entre esquinas
            if self._bfs_distance((0, 0), (self.size - 1, self.size - 1), grid=grid) is not None:
                return grid

        # fallback ultra-defensivo
        return np.zeros((self.size, self.size), dtype=np.int8)

    def _sample_start_goal_robust(
        self,
        *,
        min_bfs: int,
        max_grid_tries: int,
        max_pair_tries: int,
        regen_grid_fn: Optional[Callable[[], np.ndarray]] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Intenta samplear (start,goal) con bfs>=min_bfs.
        Si falla y max_grid_tries>1, regenera grid e intenta de nuevo.

        La regeneración se hace SOLO si regen_grid_fn != None.
        """
        min_bfs = int(max(0, min_bfs))

        for _gtry in range(max(1, int(max_grid_tries))):
            free = np.argwhere(self.grid == 0)
            if free.shape[0] < 2:
                # grid degenerado
                self.grid = np.zeros((self.size, self.size), dtype=np.int8)
                free = np.argwhere(self.grid == 0)

            for _ in range(int(max_pair_tries)):
                a = tuple(free[int(self.rng.integers(0, free.shape[0]))])
                g = tuple(free[int(self.rng.integers(0, free.shape[0]))])
                if a == g:
                    continue

                d = self._bfs_distance(a, g)
                if d is None:
                    continue
                if int(d) < int(min_bfs):
                    continue
                return a, g

            if regen_grid_fn is not None and int(max_grid_tries) > 1:
                self.grid = regen_grid_fn()

        # fallback “seguro”
        self.grid = self._generate_solvable_grid()
        a = (0, 0)
        g = (self.size - 1, self.size - 1)
        self.grid[a] = 0
        self.grid[g] = 0
        return a, g

    def _bfs_distance(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: Optional[np.ndarray] = None
    ) -> Optional[int]:
        g = self.grid if grid is None else grid

        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])

        if g[sr, sc] == 1 or g[gr, gc] == 1:
            return None
        if (sr, sc) == (gr, gc):
            return 0

        # ✅ Blindaje por si alguien cambia size/cfg: reconstruye buffers si no coinciden
        if self._bfs_dist_grid.shape != (self.size, self.size):
            self._bfs_dist_grid = np.full((self.size, self.size), -1, dtype=np.int16)
            self._bfs_queue = deque()

        dist = self._bfs_dist_grid
        dist.fill(-1)

        q = self._bfs_queue
        q.clear()

        dist[sr, sc] = 0
        q.append((sr, sc))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])

            for dr, dc in ACTIONS.values():
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                    continue
                if g[nr, nc] == 1:
                    continue
                if dist[nr, nc] != -1:
                    continue

                nd = d + 1
                if (nr, nc) == (gr, gc):
                    return nd

                dist[nr, nc] = nd
                q.append((nr, nc))

        return None