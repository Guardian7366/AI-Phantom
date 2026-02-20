from __future__ import annotations

import inspect
from typing import Dict, Any, Optional

import numpy as np


class EvaluationController:
    """
    Evaluación determinista (greedy):
    - Usa agent.act(..., deterministic=True)
    - Resetea el env con seeds distintos por episodio (reproducible con seed base)
    - Reporta métricas robustas

    Defensivo:
    - freeze_pool se pasa SOLO si env.reset() lo soporta.
    - reset_pool_on_eval permite elegir si el pool lvl2 se reinicia al inicio del batch.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self._env_reset_accepts_freeze_pool = self._fn_accepts_kw(getattr(self.env, "reset", None), "freeze_pool")

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
        if self._env_reset_accepts_freeze_pool:
            return self.env.reset(
                curriculum_level=int(curriculum_level),
                seed=int(seed),
                reset_pool=bool(reset_pool),
                freeze_pool=bool(freeze_pool),
            )
        return self.env.reset(
            curriculum_level=int(curriculum_level),
            seed=int(seed),
            reset_pool=bool(reset_pool),
        )

    def evaluate(
        self,
        episodes: int = 200,
        curriculum_level: int = 2,
        seed: int = 123,
        *,
        max_steps_override: Optional[int] = None,
        return_episode_details: bool = False,
        freeze_pool: bool = True,
        reset_pool_on_eval: bool = True,  # ✅ nuevo, no rompe (tiene default)
    ) -> Dict[str, Any]:
        episodes = int(episodes)
        if episodes <= 0:
            raise ValueError(f"episodes debe ser > 0. Recibido: {episodes}")

        lvl = int(curriculum_level)
        rng = np.random.default_rng(int(seed))

        # ✅ Resetea pool lvl2 SOLO una vez por evaluación (si aplica)
        if lvl >= 2 and bool(reset_pool_on_eval):
            _ = self._env_reset(
                curriculum_level=lvl,
                seed=int(seed),
                reset_pool=True,
                freeze_pool=bool(freeze_pool),
            )

        success_count = 0
        truncated_count = 0

        steps_all, steps_succ, steps_fail = [], [], []
        bfs_list, ratio_list = [], []

        ep_details = [] if return_episode_details else None

        # hard cap: por defecto usa cfg.max_steps del env, si existe
        env_max = int(getattr(getattr(self.env, "cfg", None), "max_steps", 0) or 0)
        hard_cap_default = env_max if env_max > 0 else 10_000
        hard_cap = int(max_steps_override) if max_steps_override is not None else int(hard_cap_default)
        if hard_cap <= 0:
            hard_cap = 10_000

        for ep in range(episodes):
            ep_seed = int(rng.integers(0, 10_000_000))

            obs, info = self._env_reset(
                curriculum_level=lvl,
                seed=ep_seed,
                reset_pool=False,
                freeze_pool=bool(freeze_pool),
            )

            done = False
            truncated = False
            steps = 0

            bfs0 = info.get("bfs_dist", None) if isinstance(info, dict) else None

            while not (done or truncated):
                a = self.agent.act(obs, deterministic=True)
                obs, _r, done, truncated, _info = self.env.step(a)
                steps += 1

                if steps >= hard_cap and not done and not truncated:
                    truncated = True
                    break

            if done:
                success_count += 1
                steps_succ.append(steps)
            else:
                steps_fail.append(steps)
                if truncated:
                    truncated_count += 1

            steps_all.append(steps)

            bfs_val = bfs0 if bfs0 is not None else steps
            bfs_list.append(int(bfs_val))
            ratio_list.append(float(steps) / float(max(1, int(bfs_val))))

            if ep_details is not None:
                ep_details.append({
                    "ep": int(ep),
                    "seed": int(ep_seed),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "steps": int(steps),
                    "bfs_dist": None if bfs0 is None else int(bfs0),
                    "ratio_vs_bfs": float(ratio_list[-1]),
                })

        sr = float(success_count) / float(episodes)

        out: Dict[str, Any] = {
            "episodes": int(episodes),
            "curriculum_level": int(lvl),
            "seed": int(seed),
            "freeze_pool": bool(freeze_pool),
            "reset_pool_on_eval": bool(reset_pool_on_eval),

            "success_rate": float(sr),
            "success_count": int(success_count),
            "fail_count": int(episodes - success_count),
            "truncated_count": int(truncated_count),

            "mean_steps": self._safe_mean(steps_all),
            "std_steps": self._safe_std(steps_all),
            "median_steps": float(np.median(steps_all)) if len(steps_all) else float("nan"),

            "mean_steps_success": self._safe_mean(steps_succ),
            "mean_steps_fail": self._safe_mean(steps_fail),

            "mean_bfs_dist": self._safe_mean(bfs_list),
            "mean_ratio_vs_bfs": self._safe_mean(ratio_list),
        }

        if len(steps_all):
            out["p90_steps"] = float(np.percentile(steps_all, 90))
            out["p99_steps"] = float(np.percentile(steps_all, 99))

        if return_episode_details:
            out["episode_details"] = ep_details

        return out