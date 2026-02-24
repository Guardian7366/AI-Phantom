# ai_phantom/envs/maze/maze_env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from collections import deque

import numpy as np

from .maze_utils import bfs_distance_map, in_bounds, manhattan, sample_free_cell

Pos = Tuple[int, int]


@dataclass
class MazeConfig:
    height: int = 12
    width: int = 12

    # Laberinto fijo (fase 0/1)
    use_walls: bool = False
    wall_prob: float = 0.25  # solo si use_walls=True

    max_steps: int = 256
    min_manhattan: int = 6  # para fase 1: evitar start/goal demasiado cerca

    # Reward shaping
    step_penalty: float = -0.01
    wall_bump_penalty: float = -0.02
    goal_reward: float = 1.0
    progress_reward: float = 0.03
    revisit_penalty: float = 0.002

    # ✅ A1: shaping independiente de la observación
    use_progress_shaping: bool = True

    # ✅ C3: shaping clamp ON por default
    progress_reward_clip: float = 0.05

    # Observación (canales)
    include_goal: bool = True
    include_visited: bool = True
    include_step_channel: bool = True

    # Exploration / anti-loop (v1 existente)
    novelty_beta: float = 0.0
    loop_terminate_visits: int = 0
    loop_penalty: float = 0.0

    # ✅ C2: Canal extra con distancia BFS al goal (normalizada)
    include_dist_channel: bool = False
    dist_invert: bool = True
    dist_clip: int = 64

    # ------------------------------
    # ✅ Sprint 2 (D): Anti-loop robusto
    # ------------------------------
    enable_loop_detection: bool = True

    # Historial de posiciones para detectar oscilación/ciclos cortos
    loop_window: int = 16           # tamaño del deque
    loop_cycle_len: int = 4         # "ciclo corto" si vuelves a una celda dentro de <=4 pasos

    # Estancamiento (no mejora BFS) por N pasos seguidos
    stagnation_steps: int = 10
    stagnation_penalty: float = 0.02

    # Penalizaciones por detección de patrones
    oscillation_penalty: float = 0.03
    short_cycle_penalty: float = 0.03

    # Terminar episodio si hay demasiadas detecciones
    terminate_on_loop: bool = True
    loop_terminate_hits: int = 3
    loop_terminate_extra_penalty: float = 0.10

    # ------------------------------
    # ✅ Diamond: Potential-based shaping (PBRS) con BFS
    # ------------------------------
    use_potential_shaping: bool = True
    potential_gamma: float = 0.99      # normalmente = PPO gamma
    potential_coef: float = 0.05       # lambda del shaping
    potential_clip: float = 0.10       # clamp del shaping (seguro)

    # Si True, desactiva el shaping "legacy" progress_reward para evitar doble señal
    disable_legacy_progress_when_potential: bool = True


class MazeEnv:
    """
    Entorno propio (sin gymnasium).
    Acciones: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Observación: np.float32 con shape (C, H, W)
    """
    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(self, config: MazeConfig, seed: int = 0):
        self.cfg = config
        self.rng = np.random.default_rng(int(seed))

        # Laberinto fijo (se construye una sola vez por instancia)
        self.walls: np.ndarray = self._build_fixed_maze()

        self.agent: Pos = (0, 0)
        self.goal: Pos = (0, 0)
        self.visited: np.ndarray = np.zeros((self.cfg.height, self.cfg.width), dtype=np.int32)
        self.t: int = 0
        self._dist_map: Optional[np.ndarray] = None

        # --- Sprint 2 (D) state ---
        self._pos_hist = deque(maxlen=max(4, int(self.cfg.loop_window)))
        self._no_progress = 0
        self._loop_hits = 0
        self._last_dist = -1

    # -------------------- Maze construction --------------------
    def _build_fixed_maze(self) -> np.ndarray:
        h, w = self.cfg.height, self.cfg.width
        walls = np.zeros((h, w), dtype=bool)

        if self.cfg.use_walls:
            p = float(self.cfg.wall_prob)
            walls = self.rng.random((h, w)) < p

            # asegurar libres (por seguridad)
            walls[0, 0] = False
            walls[h - 1, w - 1] = False

        return walls

    def rebuild_walls(self, *, seed: Optional[int] = None, wall_prob: Optional[float] = None) -> None:
        """
        ✅ C1: reconstruye el laberinto fijo (walls) de forma reproducible.
        - seed: si se pasa, reseedea el RNG interno SOLO para reconstruir walls.
        - wall_prob: si se pasa, actualiza cfg.wall_prob antes de construir.

        Nota: obs_shape NO cambia (walls channel siempre existe).
        """
        self._dist_map = None
        
        self._pos_hist.clear()
        self._no_progress = 0
        self._loop_hits = 0
        self._last_dist = -1

        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        wp = float(self.cfg.wall_prob) if wall_prob is None else float(wall_prob)
        # construye usando wp sin mutar cfg permanentemente
        old = float(self.cfg.wall_prob)
        self.cfg.wall_prob = wp
        self.walls = self._build_fixed_maze()
        self.cfg.wall_prob = old
        # No tocamos agent/goal aquí.

    # -------------------- Public API --------------------
    def reset(self, seed: Optional[int] = None, phase: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        self.t = 0
        self.visited.fill(0)

        # reset anti-loop state
        self._pos_hist.clear()
        self._no_progress = 0
        self._loop_hits = 0
        self._last_dist = -1

        if phase == 0:
            self.agent = (0, 0)
            self.goal = (self.cfg.height - 1, self.cfg.width - 1)

            if self.walls[self.agent] or self.walls[self.goal]:
                self.walls[self.agent] = False
                self.walls[self.goal] = False

            self._dist_map = bfs_distance_map(self.walls, self.goal)

        elif phase == 1:
            for _ in range(10_000):
                a = sample_free_cell(self.rng, self.walls)
                g = sample_free_cell(self.rng, self.walls)
                if a == g:
                    continue
                if manhattan(a, g) < self.cfg.min_manhattan:
                    continue

                dist = bfs_distance_map(self.walls, g)
                if dist[a[0], a[1]] == -1:
                    continue

                self.agent, self.goal = a, g
                self._dist_map = dist
                break
            else:
                raise RuntimeError(
                    "No se pudo samplear start/goal alcanzables con min_manhattan; revisa paredes/config."
                )
        else:
            raise ValueError(f"phase inválida: {phase}")

        self.visited[self.agent] += 1
        self._pos_hist.append(self.agent)

        if self._dist_map is not None:
            self._last_dist = int(self._dist_map[self.agent[0], self.agent[1]])
        else:
            self._last_dist = -1

        obs = self._make_obs()
        info = self._make_info(done=False, bumped=False, reached=False, looped=False, loop_reason=None)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        done = False
        action = int(action)
        if action not in self.ACTIONS:
            raise ValueError(f"acción inválida: {action}")

        self.t += 1

        bumped = False
        reached = False
        looped = False
        loop_reason: Optional[str] = None

        ar, ac = self.agent
        dr, dc = self.ACTIONS[action]
        nr, nc = ar + dr, ac + dc

        reward = float(self.cfg.step_penalty)

        # ✅ Control para evitar doble shaping
        if bool(self.cfg.use_potential_shaping) and bool(self.cfg.disable_legacy_progress_when_potential):
            legacy_ok = False
        else:
            legacy_ok = True

        if (not in_bounds(nr, nc, self.cfg.height, self.cfg.width)) or self.walls[nr, nc]:
            bumped = True
            reward += float(self.cfg.wall_bump_penalty)
            nr, nc = ar, ac
            # ✅ FIX: un bump NO debe contar como estancamiento acumulado
            if bool(self.cfg.enable_loop_detection):
                self._no_progress = 0   
        else:

            # ✅ Diamond: Potential-based shaping con BFS
            if (self._dist_map is not None) and bool(self.cfg.use_potential_shaping):

                d0 = int(self._dist_map[ar, ac])
                d1 = int(self._dist_map[nr, nc])

                if d0 != -1 and d1 != -1:

                    gamma_p = float(self.cfg.potential_gamma)
                    lam = float(self.cfg.potential_coef)

                    # F(s) = -d(s)
                    # gamma*F(s') - F(s) = -gamma*d1 + d0
                    shaped = lam * (float(d0) - gamma_p * float(d1))

                    clip = float(self.cfg.potential_clip)
                    if clip > 0.0:
                        shaped = float(np.clip(shaped, -clip, clip))

                    reward += shaped

            # ✅ Legacy shaping (solo si permitido)
            elif legacy_ok and self.cfg.use_progress_shaping and (self._dist_map is not None):

                d0 = int(self._dist_map[ar, ac])
                d1 = int(self._dist_map[nr, nc])

                if d0 != -1 and d1 != -1:
                    delta = d0 - d1
                    if delta > 0:
                        shaped = float(self.cfg.progress_reward) * float(delta)

                        clip = float(self.cfg.progress_reward_clip)
                        if clip > 0.0:
                            shaped = float(np.clip(shaped, -clip, clip))

                        reward += shaped

        prev_pos = self.agent
        self.agent = (nr, nc)

        moved = (self.agent != prev_pos)

        # revisit penalty solo si se movió
        if moved and self.visited[self.agent] > 0:
            reward -= float(self.cfg.revisit_penalty)

        # ----- intrinsic novelty + legacy anti-loop by visits -----
        if self.cfg.include_visited:
            n = int(self.visited[self.agent])  # visitas ANTES de incrementar

            if self.cfg.novelty_beta > 0.0:
                reward += float(self.cfg.novelty_beta) / float(np.sqrt(n + 1))

            if self.cfg.loop_terminate_visits and n >= int(self.cfg.loop_terminate_visits):
                done = True
                reached = False
                reward -= float(self.cfg.loop_penalty)

        # ------------------------------
        # ✅ Sprint 2 (D): loop detection
        # ------------------------------
        if bool(self.cfg.enable_loop_detection) and moved:
            # actualizar historial solo si hubo movimiento real
            self._pos_hist.append(self.agent)

            # 1) Estancamiento por BFS (no mejora distancia)
            if self._dist_map is not None:
                d_now = int(self._dist_map[self.agent[0], self.agent[1]])
                d_prev = int(self._dist_map[prev_pos[0], prev_pos[1]])
                # si es inválida (-1), no contamos estancamiento
                if d_prev != -1 and d_now != -1:
                    # ✅ No cuentes bumps como “no-progress” (evita matar episodios rescatables)
                    if bumped:
                        self._no_progress = 0
                    else:
                        if d_now >= d_prev:
                            self._no_progress += 1
                        else:
                            self._no_progress = 0
                else:
                    self._no_progress = 0

                if self._no_progress >= int(self.cfg.stagnation_steps):
                    looped = True
                    loop_reason = "stagnation"
                    reward -= float(self.cfg.stagnation_penalty)

            # 2) Oscilación A<->B<->A
            if len(self._pos_hist) >= 3:
                p3 = self._pos_hist[-3]
                p2 = self._pos_hist[-2]
                p1 = self._pos_hist[-1]
                if (p1 == p3) and (p2 != p1):
                    looped = True
                    loop_reason = loop_reason or "oscillation"
                    reward -= float(self.cfg.oscillation_penalty)

            # 3) Ciclo corto (volver a una celda dentro de K pasos)
            k = int(self.cfg.loop_cycle_len)
            if k > 1 and len(self._pos_hist) >= (k + 1):
                # revisa si la posición actual aparece en las últimas k posiciones previas
                recent_prev = list(self._pos_hist)[-(k + 1):-1]
                if self.agent in recent_prev:
                    # si hubo mejora de distancia este paso, no castigues short_cycle
                    improved = False
                    if self._dist_map is not None:
                        d_now = int(self._dist_map[self.agent[0], self.agent[1]])
                        d_prev = int(self._dist_map[prev_pos[0], prev_pos[1]])
                        if d_prev != -1 and d_now != -1 and d_now < d_prev:
                            improved = True

                    if not improved:
                        looped = True
                        loop_reason = loop_reason or "short_cycle"
                        reward -= float(self.cfg.short_cycle_penalty)

            if looped:
                self._loop_hits += 1
                if bool(self.cfg.terminate_on_loop) and self._loop_hits >= int(self.cfg.loop_terminate_hits):
                    done = True
                    reached = False
                    reward -= float(self.cfg.loop_terminate_extra_penalty)

        # visited solo si se movió
        if moved:
            self.visited[self.agent] += 1

        if self.agent == self.goal:
            reached = True
            reward += float(self.cfg.goal_reward)

        done = bool(done or reached or (self.t >= self.cfg.max_steps))

        obs = self._make_obs()
        info = self._make_info(done=done, bumped=bumped, reached=reached, looped=looped, loop_reason=loop_reason)
        return obs, float(reward), bool(done), info

    def render(self, mode: str = "ansi") -> str:
        if mode != "ansi":
            raise ValueError("Por ahora solo soporta mode='ansi'.")

        h, w = self.cfg.height, self.cfg.width
        rows = []
        for r in range(h):
            line = []
            for c in range(w):
                if self.walls[r, c]:
                    ch = "#"
                elif (r, c) == self.agent:
                    ch = "A"
                elif (r, c) == self.goal:
                    ch = "G"
                else:
                    ch = "."
                line.append(ch)
            rows.append("".join(line))
        return "\n".join(rows)

    # -------------------- Internals --------------------
    def _make_info(
        self,
        done: bool,
        bumped: bool,
        reached: bool,
        looped: bool,
        loop_reason: Optional[str],
    ) -> Dict[str, Any]:
        term_reason = None
        if bool(reached):
            term_reason = "reached"
        elif bool(done):
            # si terminó y no llegó
            if bool(looped) and int(self._loop_hits) >= 1 and bool(self.cfg.terminate_on_loop):
                term_reason = "loop"
            elif self.t >= int(self.cfg.max_steps):
                term_reason = "timeout"
            else:
                term_reason = "other"

        return {
            "t": self.t,
            "agent": self.agent,
            "goal": self.goal,
            "done": done,
            "bumped": bumped,
            "reached": reached,
            "looped": bool(looped),
            "loop_reason": loop_reason,
            "no_progress": int(self._no_progress),
            "loop_hits": int(self._loop_hits),
            "term_reason": term_reason,   # ✅ NUEVO
        }

    def _make_obs(self) -> np.ndarray:
        """
        Canales:
          0: walls
          1: agent
          2: goal (opcional)
          3: visited (opcional)
          4: step progress (opcional)
          +: dist-to-goal (opcional, ✅ C2)
        """
        h, w = self.cfg.height, self.cfg.width
        channels = []

        walls = self.walls.astype(np.float32)
        channels.append(walls)

        a = np.zeros((h, w), dtype=np.float32)
        a[self.agent] = 1.0
        channels.append(a)

        if self.cfg.include_goal:
            g = np.zeros((h, w), dtype=np.float32)
            g[self.goal] = 1.0
            channels.append(g)

        if self.cfg.include_visited:
            v = self.visited.astype(np.float32)
            v = np.clip(v, 0.0, 10.0) / 10.0
            channels.append(v)

        if self.cfg.include_step_channel:
            s = np.full((h, w), fill_value=float(self.t) / float(self.cfg.max_steps), dtype=np.float32)
            channels.append(s)

        if bool(self.cfg.include_dist_channel):
            if self._dist_map is None:
                d = np.ones((h, w), dtype=np.float32)
            else:
                dm = self._dist_map.astype(np.float32)

                unreachable = (dm < 0)
                dm = dm.copy()
                dm[unreachable] = 0.0

                clip = int(self.cfg.dist_clip)
                if clip > 0:
                    dm = np.clip(dm, 0.0, float(clip))

                dmax = float(dm.max())
                if dmax < 1e-6:
                    dn = np.zeros_like(dm, dtype=np.float32)
                else:
                    dn = dm / dmax

                dn[unreachable] = 1.0

                if bool(self.cfg.dist_invert):
                    dn = 1.0 - dn

                d = dn.astype(np.float32)

            channels.append(d)

        obs = np.stack(channels, axis=0).astype(np.float32)
        return obs