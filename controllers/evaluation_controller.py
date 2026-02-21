# controllers/evaluation_controller.py
from __future__ import annotations
import zlib
import inspect
from typing import Dict, Any, Optional, List, Tuple

import numpy as np


class EvaluationController:
    """
    Evaluación determinista (greedy):
    - Usa agent.act(..., deterministic=True) si existe
    - Resetea el env con seeds distintos por episodio
    - Reporta métricas robustas y alineadas con entrenamiento

    Defensivo:
    - freeze_pool se pasa SOLO si env.reset() lo soporta.
    - reset_pool_on_eval permite elegir si el pool lvl2 se reinicia al inicio del batch.
    - record_trajectories y track_grid_hashes son opcionales.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self._env_reset_accepts_freeze_pool = self._fn_accepts_kw(getattr(self.env, "reset", None), "freeze_pool")
        self._env_reset_accepts_reset_pool = self._fn_accepts_kw(getattr(self.env, "reset", None), "reset_pool")

        # Agent act signature
        self._agent_act_accepts_deterministic = self._fn_accepts_kw(getattr(self.agent, "act", None), "deterministic")

    @staticmethod
    def _fn_accepts_kw(fn, kw: str) -> bool:
        if fn is None:
            return False
        try:
            sig = inspect.signature(fn)
            return kw in sig.parameters
        except Exception:
            return False

    @staticmethod
    def _safe_mean(x) -> float:
        return float(np.mean(x)) if len(x) else float("nan")

    @staticmethod
    def _safe_std(x) -> float:
        return float(np.std(x)) if len(x) else float("nan")

    def _env_reset(self, *, curriculum_level: int, seed: int, reset_pool: bool, freeze_pool: bool):
        kwargs = {
            "curriculum_level": int(curriculum_level),
            "seed": int(seed),
        }
        if self._env_reset_accepts_reset_pool:
            kwargs["reset_pool"] = bool(reset_pool)
        if self._env_reset_accepts_freeze_pool:
            kwargs["freeze_pool"] = bool(freeze_pool)
        return self.env.reset(**kwargs)

    def _agent_eval_mode(self):
        # Compatible DQN/PPO (y futuros)
        if hasattr(self.agent, "eval_mode"):
            try:
                self.agent.eval_mode()
                return
            except Exception:
                pass
        # fallback: si expone .q o .net
        for attr in ("q", "net", "model"):
            m = getattr(self.agent, attr, None)
            if hasattr(m, "eval"):
                try:
                    m.eval()
                except Exception:
                    pass

    @staticmethod
    def _maybe_hash_grid(env) -> Optional[int]:
        grid = getattr(env, "grid", None)
        if grid is None:
            return None
        try:
            arr = np.asarray(grid)
            if arr.ndim != 2:
                return None
            return int(zlib.crc32(arr.tobytes()) & 0xFFFFFFFF)
        except Exception:
            return None

    @staticmethod
    def _coerce_int(v, default: int = 0) -> int:
        try:
            if v is None:
                return int(default)
            return int(v)
        except Exception:
            return int(default)

    def _act_deterministic(self, obs: np.ndarray) -> int:
        if self._agent_act_accepts_deterministic:
            return int(self.agent.act(obs, deterministic=True))
        # fallback: si no soporta deterministic, llama sin kw
        return int(self.agent.act(obs))

    def evaluate(
        self,
        episodes: int = 200,
        curriculum_level: int = 2,
        seed: int = 123,
        *,
        max_steps_override: Optional[int] = None,
        return_episode_details: bool = False,
        freeze_pool: bool = True,
        reset_pool_on_eval: bool = True,
        episode_seeds: Optional[List[int]] = None,
        record_trajectories: bool = False,
        track_grid_hashes: bool = False,
    ) -> Dict[str, Any]:
        episodes = int(episodes)
        if episodes <= 0:
            raise ValueError(f"episodes debe ser > 0. Recibido: {episodes}")

        lvl = int(curriculum_level)

        # agente en eval (seguridad)
        self._agent_eval_mode()

        # Seeds por episodio
        if episode_seeds is not None:
            if len(episode_seeds) < episodes:
                raise ValueError(
                    f"episode_seeds tiene {len(episode_seeds)} seeds pero episodes={episodes}. "
                    f"Deben coincidir o episode_seeds debe ser >= episodes."
                )
            ep_seeds = [int(s) for s in episode_seeds[:episodes]]
        else:
            rng = np.random.default_rng(int(seed))
            ep_seeds = [int(rng.integers(0, 10_000_000)) for _ in range(episodes)]

        # reset pool una vez por batch
        if lvl >= 2 and bool(reset_pool_on_eval):
            _ = self._env_reset(
                curriculum_level=lvl,
                seed=int(seed),
                reset_pool=True,
                freeze_pool=bool(freeze_pool),
            )

        env_max = int(getattr(getattr(self.env, "cfg", None), "max_steps", 0) or 0)
        hard_cap_default = env_max if env_max > 0 else 10_000
        hard_cap = int(max_steps_override) if max_steps_override is not None else int(hard_cap_default)
        if hard_cap <= 0:
            hard_cap = 10_000

        success_count = 0
        truncated_count = 0

        steps_all: List[int] = []
        steps_succ: List[int] = []
        steps_fail: List[int] = []

        bfs0_list: List[int] = []
        ratio_list: List[float] = []
        ratio_succ: List[float] = []
        ratio_fail: List[float] = []

        invalid_moves_list: List[int] = []
        revisit_steps_list: List[int] = []
        visited_unique_list: List[int] = []

        ep_details = [] if return_episode_details else None
        unique_grids = set() if track_grid_hashes else None

        for ep in range(episodes):
            ep_seed = int(ep_seeds[ep])

            obs, info = self._env_reset(
                curriculum_level=lvl,
                seed=ep_seed,
                reset_pool=False,
                freeze_pool=bool(freeze_pool),
            )

            if unique_grids is not None:
                h = self._maybe_hash_grid(self.env)
                if h is not None:
                    unique_grids.add(h)

            done = False
            truncated = False
            steps = 0
            reached_goal = False

            bfs0 = None
            if isinstance(info, dict):
                bfs0 = info.get("bfs_dist_start", None)
                if bfs0 is None:
                    bfs0 = info.get("bfs_dist", None)

            inv_moves = self._coerce_int(info.get("invalid_moves", 0) if isinstance(info, dict) else 0, 0)
            rev_steps = self._coerce_int(info.get("revisit_steps", 0) if isinstance(info, dict) else 0, 0)
            vis_unique = self._coerce_int(info.get("visited_unique", 0) if isinstance(info, dict) else 0, 0)

            traj: Optional[List[Tuple[int, int]]] = [] if (record_trajectories and ep_details is not None) else None
            if traj is not None and isinstance(info, dict) and "agent_pos" in info:
                try:
                    ap = info["agent_pos"]
                    traj.append((int(ap[0]), int(ap[1])))
                except Exception:
                    pass

            while not (done or truncated):
                a = self._act_deterministic(obs)
                obs, _r, done, truncated, step_info = self.env.step(a)
                steps += 1

                if isinstance(step_info, dict):
                    reached_goal = bool(step_info.get("reached_goal", reached_goal))
                    inv_moves = self._coerce_int(step_info.get("invalid_moves", inv_moves), inv_moves)
                    rev_steps = self._coerce_int(step_info.get("revisit_steps", rev_steps), rev_steps)
                    vis_unique = self._coerce_int(step_info.get("visited_unique", vis_unique), vis_unique)

                if traj is not None and isinstance(step_info, dict) and "agent_pos" in step_info:
                    try:
                        ap = step_info["agent_pos"]
                        traj.append((int(ap[0]), int(ap[1])))
                    except Exception:
                        pass

                if steps >= hard_cap and not done and not truncated:
                    truncated = True
                    break

            steps_all.append(int(steps))

            bfs0_int = self._coerce_int(bfs0, default=int(steps))
            if bfs0_int <= 0:
                bfs0_int = int(steps)
            bfs0_list.append(int(bfs0_int))

            ratio = float(steps) / float(max(1, int(bfs0_int)))
            ratio_list.append(float(ratio))

            invalid_moves_list.append(int(inv_moves))
            revisit_steps_list.append(int(rev_steps))
            visited_unique_list.append(int(vis_unique))

            if reached_goal:
                success_count += 1
                steps_succ.append(int(steps))
                ratio_succ.append(float(ratio))
            else:
                steps_fail.append(int(steps))
                ratio_fail.append(float(ratio))
                if truncated:
                    truncated_count += 1

            if ep_details is not None:
                ep_details.append({
                    "ep": int(ep),
                    "seed": int(ep_seed),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "steps": int(steps),
                    "bfs_dist_start": None if bfs0 is None else int(bfs0_int),
                    "ratio_vs_bfs_start": float(ratio),
                    "invalid_moves": int(inv_moves),
                    "revisit_steps": int(rev_steps),
                    "visited_unique": int(vis_unique),
                    **({"trajectory": traj} if traj is not None else {}),
                    **({"grid_hash": self._maybe_hash_grid(self.env)} if track_grid_hashes else {}),
                })

        sr = float(success_count) / float(episodes)

        out: Dict[str, Any] = {
            "episodes": int(episodes),
            "curriculum_level": int(lvl),
            "seed": int(seed),
            "freeze_pool": bool(freeze_pool),
            "reset_pool_on_eval": bool(reset_pool_on_eval),
            "max_steps_used": int(hard_cap),
            "env_max_steps": int(env_max),
            "success_rate": float(sr),
            "success_count": int(success_count),
            "fail_count": int(episodes - success_count),
            "truncated_count": int(truncated_count),

            "mean_steps": self._safe_mean(steps_all),
            "std_steps": self._safe_std(steps_all),
            "median_steps": float(np.median(steps_all)) if len(steps_all) else float("nan"),

            "mean_steps_success": self._safe_mean(steps_succ),
            "mean_steps_fail": self._safe_mean(steps_fail),

            "mean_bfs_dist_start": self._safe_mean(bfs0_list),
            "mean_ratio_vs_bfs_start": self._safe_mean(ratio_list),

            "mean_ratio_vs_bfs_start_success": self._safe_mean(ratio_succ),
            "mean_ratio_vs_bfs_start_fail": self._safe_mean(ratio_fail),

            "mean_invalid_moves": self._safe_mean(invalid_moves_list),
            "mean_revisit_steps": self._safe_mean(revisit_steps_list),
            "mean_visited_unique": self._safe_mean(visited_unique_list),
        }

        if len(steps_all):
            out["p90_steps"] = float(np.percentile(steps_all, 90))
            out["p99_steps"] = float(np.percentile(steps_all, 99))

        if len(visited_unique_list):
            out["p10_visited_unique"] = float(np.percentile(visited_unique_list, 10))
            out["p50_visited_unique"] = float(np.percentile(visited_unique_list, 50))

        if track_grid_hashes and unique_grids is not None:
            out["unique_grid_count"] = int(len(unique_grids))

        if return_episode_details:
            out["episode_details"] = ep_details

        return out