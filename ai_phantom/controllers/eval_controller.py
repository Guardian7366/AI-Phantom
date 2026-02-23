# ai_phantom/controllers/eval_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch


@dataclass
class EvalConfig:
    episodes: int
    phase: int

    seed_base: int = 10_000
    deterministic: bool = True

    # ✅ Multi-walls eval (clave)
    rebuild_walls_each_episode: bool = False
    walls_seed_base: int = 777
    wall_prob: Optional[float] = None


class EvalController:
    def __init__(self, env, policy, device):
        self.env = env
        self.policy = policy
        self.device = device

    @torch.no_grad()
    def evaluate(self, cfg: EvalConfig) -> Dict[str, Any]:
        successes = 0
        steps_sum = 0

        max_steps: Optional[int] = None
        if hasattr(self.env, "cfg") and getattr(self.env, "cfg") is not None:
            max_steps = getattr(self.env.cfg, "max_steps", None)

        for ep in range(int(cfg.episodes)):
            seed = int(cfg.seed_base + ep)

            # ✅ rebuild walls por episodio (si el env lo soporta)
            if bool(cfg.rebuild_walls_each_episode) and hasattr(self.env, "rebuild_walls"):
                ws = int(cfg.walls_seed_base + ep)
                self.env.rebuild_walls(seed=ws, wall_prob=cfg.wall_prob)

            if cfg.deterministic:
                self.policy.set_deterministic_seed(seed)

            obs, info = self.env.reset(seed=seed, phase=int(cfg.phase))

            done = False
            steps = 0
            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()
                out = self.policy.act(obs_t, deterministic=bool(cfg.deterministic))
                a = int(out.action.item())

                obs, r, done, info = self.env.step(a)
                steps += 1

            steps_sum += steps
            if info.get("reached", False):
                successes += 1

        if cfg.deterministic:
            self.policy.set_deterministic_seed(None)

        sr = successes / float(cfg.episodes) if cfg.episodes > 0 else 0.0
        avg_steps = steps_sum / float(cfg.episodes) if cfg.episodes > 0 else 0.0
        return {"sr": float(sr), "avg_steps": float(avg_steps)}