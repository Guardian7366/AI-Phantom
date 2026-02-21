# environments/maze/maze_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, Callable
from collections import deque

import numpy as np

from environments.maze.maze_families import FAMILY_REGISTRY

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

    # “anti-loop (ACTIVO por defecto en esta fase; poner 0.0 para apagar)”
    revisit_penalty: float = 0.0015
    new_cell_bonus: float = 0.0005

    # Evita casos triviales / demasiado cortos
    min_bfs_start_goal_lvl1: int = 2
    min_bfs_start_goal_lvl2: int = 4

    # Pool de grids para lvl2 (estabilidad + generalización)
    lvl2_grid_pool_size: int = 64
    lvl2_pool_refresh_prob: float = 0.05

    # Mixture curriculum (familias de laberintos)
    # Si es None o families.enabled=false => comportamiento actual (backward compatible)
    families: Optional[Dict[str, Any]] = None


class MazeEnvironment:
    """
    Laberinto NxN con muros (1=wall, 0=free). Genera laberintos resolubles.
    Observación: (3, H, W) = (walls, agent, goal).

    Reproducibilidad / estabilidad:
    - reset(seed=...) reseedea el RNG del env para reproducibilidad por episodio.
    - En lvl2 con pool activo, el pool SOLO se resetea si reset_pool=True (y lvl>=2).
    - freeze_pool=True evita refreshes del pool durante evaluación/batch (señal comparable).

    PERF (FASE 2.9.x):
    - step() NO hace BFS. Se precomputa un "distance map" desde la meta una vez por episodio.
    - progress reward usa delta_dist = last_dist - new_dist (O(1)).

    NUEVO (FASE 2.9.10+):
    - Sampling start/goal SIN BFS repetitivo: elige goal -> construye dist_map una vez -> samplea start.
    - Familias de laberintos (mixture curriculum) integradas (especialmente lvl2).
    """

    def __init__(self, config: Optional[MazeConfig] = None, rng_seed: int = 0):
        self.cfg = config or MazeConfig()
        self.rng = np.random.default_rng(int(rng_seed))

        self.size = int(self.cfg.size)
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (self.size - 1, self.size - 1)

        self.steps = 0

        # Distancia actual desde agente a meta según goal distance map.
        self.last_bfs_dist: Optional[int] = None
        self.start_bfs_dist: Optional[int] = None

        # Nivel 0/1: grid fijo
        self.fixed_grid_cache: Optional[np.ndarray] = None

        # Pool lvl2
        self._lvl2_pool: List[np.ndarray] = []
        self._lvl2_pool_size = max(0, int(getattr(self.cfg, "lvl2_grid_pool_size", 0)))
        self._lvl2_refresh_prob = float(getattr(self.cfg, "lvl2_pool_refresh_prob", 0.0))

        # -------------------------
        # Mixture curriculum (familias)
        # -------------------------
        self._families_cfg = getattr(self.cfg, "families", None)
        self._families_enabled = False
        self._family_params: Dict[str, Dict[str, Any]] = {}
        self._mixture_by_level: Dict[int, Dict[str, float]] = {}
        self._last_family: str = "bernoulli"

        if isinstance(self._families_cfg, dict) and bool(self._families_cfg.get("enabled", False)):
            self._families_enabled = True

            reg = self._families_cfg.get("registry", {})
            if isinstance(reg, dict):
                for k, v in reg.items():
                    if isinstance(v, dict):
                        self._family_params[str(k)] = dict(v)

            mix = self._families_cfg.get("mixture", {})
            if isinstance(mix, dict):
                for lvl_key, dist in mix.items():
                    try:
                        lvl = int(str(lvl_key).replace("lvl", "").strip())
                    except Exception:
                        continue
                    if isinstance(dist, dict):
                        dd = {
                            str(n): float(w)
                            for n, w in dist.items()
                            if str(n) in FAMILY_REGISTRY and float(w) > 0.0
                        }
                        if dd:
                            self._mixture_by_level[int(lvl)] = dd

        # BFS buffers reutilizables (para _bfs_distance solo en generación/fallback)
        self._bfs_dist_grid = np.full((self.size, self.size), -1, dtype=np.int16)
        self._bfs_queue: deque = deque()

        # Distance map reutilizable (desde goal) -> O(1) en step()
        self._goal_dist_grid = np.full((self.size, self.size), -1, dtype=np.int16)
        self._goal_bfs_queue: deque = deque()

        # Tracking episodio: visitas (anti-loop + métricas)
        self._visit_count = np.zeros((self.size, self.size), dtype=np.int16)
        self._visited_unique = 0
        self._revisit_steps = 0
        self._invalid_moves = 0

        # Anti-estancamiento (barato, reproducible)
        self._no_progress_steps = 0
        self._best_bfs_dist: Optional[int] = None

        # bandera para evitar recomputar distmap si ya se hizo en sampling
        self._distmap_ready_this_reset = False

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
        self._last_family = "bernoulli"

        # Re-seed del RNG (reproducible por episodio)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        # Blindaje: solo lvl2 puede resetear pool
        if bool(reset_pool) and lvl >= 2 and self._lvl2_pool_size > 0:
            self._lvl2_pool.clear()

        self.steps = 0
        self.start_bfs_dist = None
        self.last_bfs_dist = None

        # reset métricas episodio
        if self._visit_count.shape != (self.size, self.size):
            self._visit_count = np.zeros((self.size, self.size), dtype=np.int16)
        else:
            self._visit_count.fill(0)
        self._visited_unique = 0
        self._revisit_steps = 0
        self._invalid_moves = 0

        self._no_progress_steps = 0
        self._best_bfs_dist = None
        self._distmap_ready_this_reset = False

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
                regen_grid_fn=None,
            )
            self.grid[self.agent_pos] = 0
            self.grid[self.goal_pos] = 0

        else:
            # lvl2: grid variable + start/goal aleatorios (pool si activo)
            min_dist = int(getattr(self.cfg, "min_bfs_start_goal_lvl2", 0))

            def _lvl2_grid() -> np.ndarray:
                # ✅ familia determinista por episodio (por seed)
                self._last_family = self._pick_family_for_level(lvl)
                return self._sample_or_build_lvl2_grid(
                    allow_refresh=(not bool(freeze_pool)),
                    family_name=self._last_family,
                )

            self.grid = _lvl2_grid()

            self.agent_pos, self.goal_pos = self._sample_start_goal_robust(
                min_bfs=min_dist,
                max_grid_tries=30,
                max_pair_tries=500,
                regen_grid_fn=_lvl2_grid,
            )
            self.grid[self.agent_pos] = 0
            self.grid[self.goal_pos] = 0

        # construir distance map 1 vez por episodio (desde goal_pos)
        if not bool(self._distmap_ready_this_reset):
            self._build_goal_distance_map(goal=self.goal_pos)
            self._distmap_ready_this_reset = True

        # dist inicial (lookup O(1))
        self.last_bfs_dist = self._lookup_goal_dist(self.agent_pos)
        self.start_bfs_dist = self.last_bfs_dist
        self._best_bfs_dist = self.last_bfs_dist

        # marcar visita inicial
        self._mark_visit(self.agent_pos)

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "bfs_dist": self.last_bfs_dist,
            "bfs_dist_start": self.start_bfs_dist,
            "curriculum_level": lvl,
            "freeze_pool": bool(freeze_pool),
            "family": str(self._last_family),  # ✅ clave para depurar colapsos por familia
            "visited_unique": int(self._visited_unique),
            "revisit_steps": int(self._revisit_steps),
            "invalid_moves": int(self._invalid_moves),
            "grid_checksum": int(self.grid.sum()),
            "grid_checksum2": int((self.grid * np.arange(1, self.size*self.size+1, dtype=np.int32).reshape(self.size, self.size)).sum()),
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

        # movimiento
        if invalid:
            r += float(self.cfg.invalid_move_penalty)
            self._invalid_moves += 1
            new_dist: Optional[int] = self.last_bfs_dist
        else:
            self.agent_pos = (nr, nc)
            new_dist = self._lookup_goal_dist(self.agent_pos)

        # anti-loop opcional
        revisit_pen = float(getattr(self.cfg, "revisit_penalty", 0.0))
        new_cell_bonus = float(getattr(self.cfg, "new_cell_bonus", 0.0))
        if (revisit_pen != 0.0) or (new_cell_bonus != 0.0):
            was_new = self._mark_visit(self.agent_pos)
            if was_new:
                if new_cell_bonus != 0.0:
                    r += new_cell_bonus
            else:
                if revisit_pen != 0.0:
                    rr, cc = int(self.agent_pos[0]), int(self.agent_pos[1])
                    v = int(self._visit_count[rr, cc])
                    r -= revisit_pen * float(max(1, v - 1))
        else:
            self._mark_visit(self.agent_pos)

        # progress reward por delta de distancia (O(1))
        if self.last_bfs_dist is not None and new_dist is not None:
            delta = int(self.last_bfs_dist) - int(new_dist)
            if delta > 0:
                r += float(self.cfg.progress_reward) * float(delta)

        # anti-estancamiento ligado a revisit_penalty (si está activo)
        if new_dist is not None:
            if (self._best_bfs_dist is None) or (int(new_dist) < int(self._best_bfs_dist)):
                self._best_bfs_dist = int(new_dist)
                self._no_progress_steps = 0
            else:
                self._no_progress_steps += 1

            if revisit_pen > 0.0 and self._no_progress_steps >= 12:
                r -= revisit_pen

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
            "bfs_dist_start": self.start_bfs_dist,
            "steps": int(self.steps),
            "invalid": bool(invalid),
            "visited_unique": int(self._visited_unique),
            "revisit_steps": int(self._revisit_steps),
            "invalid_moves": int(self._invalid_moves),
            "family": str(self._last_family),
            "reached_goal": bool(terminated),
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
    # Tracking visitas (episodio)
    # -------------------------
    def _mark_visit(self, pos: Tuple[int, int]) -> bool:
        """Retorna True si es celda nueva (primera vez), False si revisit."""
        r, c = int(pos[0]), int(pos[1])
        prev = int(self._visit_count[r, c])
        self._visit_count[r, c] = prev + 1
        if prev == 0:
            self._visited_unique += 1
            return True
        self._revisit_steps += 1
        return False

    # -------------------------
    # Distance map
    # -------------------------
    def _lookup_goal_dist(self, pos: Tuple[int, int]) -> Optional[int]:
        r, c = int(pos[0]), int(pos[1])
        d = int(self._goal_dist_grid[r, c])
        return None if d < 0 else d

    def _build_goal_distance_map(self, *, goal: Tuple[int, int]) -> None:
        if self._goal_dist_grid.shape != (self.size, self.size):
            self._goal_dist_grid = np.full((self.size, self.size), -1, dtype=np.int16)
            self._goal_bfs_queue = deque()

        dist = self._goal_dist_grid
        dist.fill(-1)

        q = self._goal_bfs_queue
        q.clear()

        gr, gc = int(goal[0]), int(goal[1])

        if self.grid[gr, gc] == 1:
            return

        dist[gr, gc] = 0
        q.append((gr, gc))

        while q:
            r, c = q.popleft()
            d = int(dist[r, c])

            for dr, dc in ACTIONS.values():
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
                    continue
                if self.grid[nr, nc] == 1:
                    continue
                if dist[nr, nc] != -1:
                    continue
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    # -------------------------
    # Mixture helpers
    # -------------------------
    def _pick_family_for_level(self, lvl: int) -> str:
        if not bool(self._families_enabled):
            return "bernoulli"

        dist = self._mixture_by_level.get(int(lvl), None)
        if not dist:
            return "bernoulli"

        names = list(dist.keys())
        weights = np.array([dist[n] for n in names], dtype=np.float64)
        s = float(weights.sum())
        if not np.isfinite(s) or s <= 0.0:
            return "bernoulli"
        probs = weights / s
        idx = int(self.rng.choice(len(names), p=probs))
        return str(names[idx])

    def _generate_grid_from_family(self, family_name: str) -> np.ndarray:
        name = str(family_name)
        gen = FAMILY_REGISTRY.get(name, None)
        if gen is None:
            name = "bernoulli"
            gen = FAMILY_REGISTRY[name]

        base = {
            "size": int(self.size),
            "wall_prob": float(getattr(self.cfg, "wall_prob", 0.28)),
        }
        extra = dict(self._family_params.get(name, {}))
        params = {**base, **extra}
        return gen(self.rng, **params)

    # -------------------------
    # Lvl2 pool helpers
    # -------------------------
    def _sample_or_build_lvl2_grid(
        self,
        force_rebuild: bool = False,
        allow_refresh: bool = True,
        *,
        family_name: Optional[str] = None,
    ) -> np.ndarray:
        pool_size = int(self._lvl2_pool_size)
        fam = str(family_name) if family_name is not None else "bernoulli"

        if pool_size <= 0:
            return self._generate_solvable_grid(family_name=fam)

        if bool(force_rebuild):
            self._lvl2_pool.clear()

        if len(self._lvl2_pool) == 0:
            for _ in range(pool_size):
                self._lvl2_pool.append(self._generate_solvable_grid(family_name=fam))

        if bool(allow_refresh) and (self.rng.random() < float(self._lvl2_refresh_prob)):
            i = int(self.rng.integers(0, len(self._lvl2_pool)))
            self._lvl2_pool[i] = self._generate_solvable_grid(family_name=fam)

        j = int(self.rng.integers(0, len(self._lvl2_pool)))
        return self._lvl2_pool[j].copy()

    def _generate_solvable_grid(self, *, family_name: Optional[str] = None) -> np.ndarray:
        fam = str(family_name) if family_name is not None else "bernoulli"

        for _ in range(int(self.cfg.max_gen_tries)):
            if bool(self._families_enabled):
                grid = self._generate_grid_from_family(fam)
            else:
                grid = self._generate_grid_from_family("bernoulli")

            if self._bfs_distance((0, 0), (self.size - 1, self.size - 1), grid=grid) is not None:
                return grid

        return np.zeros((self.size, self.size), dtype=np.int8)

    # -------------------------
    # Sampling start/goal rápido
    # -------------------------
    def _sample_start_goal_robust(
        self,
        *,
        min_bfs: int,
        max_grid_tries: int,
        max_pair_tries: int,
        regen_grid_fn: Optional[Callable[[], np.ndarray]] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        min_bfs = int(max(0, min_bfs))

        for _gtry in range(max(1, int(max_grid_tries))):
            free = np.argwhere(self.grid == 0)
            if free.shape[0] < 2:
                self.grid = np.zeros((self.size, self.size), dtype=np.int8)
                free = np.argwhere(self.grid == 0)

            for _ in range(int(max_pair_tries)):
                gi = int(self.rng.integers(0, free.shape[0]))
                gr, gc = int(free[gi, 0]), int(free[gi, 1])
                goal = (gr, gc)

                self._build_goal_distance_map(goal=goal)
                self._distmap_ready_this_reset = True
                dist = self._goal_dist_grid

                mask = dist >= min_bfs
                mask[gr, gc] = False

                candidates = np.argwhere(mask)
                if candidates.shape[0] == 0:
                    continue

                ai = int(self.rng.integers(0, candidates.shape[0]))
                ar, ac = int(candidates[ai, 0]), int(candidates[ai, 1])
                start = (ar, ac)

                return start, goal

            if regen_grid_fn is not None and int(max_grid_tries) > 1:
                self.grid = regen_grid_fn()

        # fallback seguro
        self.grid = self._generate_solvable_grid()
        a = (0, 0)
        g = (self.size - 1, self.size - 1)
        self.grid[a] = 0
        self.grid[g] = 0
        return a, g

    # -------------------------
    # BFS puntual (solo para generación/fallback)
    # -------------------------
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