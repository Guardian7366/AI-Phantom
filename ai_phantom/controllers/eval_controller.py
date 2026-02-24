# ai_phantom/controllers/eval_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from ai_phantom.agents.ppo.logits_utils import sanitize_logits_keep_neginf
from ai_phantom.planners.bfs import bfs_plan, path_to_actions
import torch.nn.functional as F
from ai_phantom.agents.ppo.action_mask import mask_invalid_actions


@dataclass
class EvalConfig:
    episodes: int
    phase: int

    seed_base: int = 10_000
    deterministic: bool = True

    rebuild_walls_each_episode: bool = False
    walls_seed_base: int = 777
    wall_prob: Optional[float] = None

    # ✅ Hybrid controller
    hybrid: bool = False
    hybrid_on_loop: bool = True
    hybrid_min_conf: float = 0.55
    hybrid_min_margin: float = 0.08  # NUEVO: diferencia top1-top2


class EvalController:
    def __init__(self, env, policy, device):
        self.env = env
        self.policy = policy
        self.device = device

    @torch.no_grad()
    def evaluate(self, cfg: EvalConfig) -> Dict[str, Any]:
        successes = 0
        steps_sum = 0
        bfs_used = 0

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

                # 1) acción PPO
                out = self.policy.act(obs_t, deterministic=bool(cfg.deterministic))
                a = int(out.action.item())

                use_bfs = False

                if bool(cfg.hybrid):

                    # A) loop detectado por env
                    if bool(cfg.hybrid_on_loop) and bool(info.get("looped", False)):
                        use_bfs = True

                    # B) baja confianza + bajo margen
                    with torch.no_grad():
                        self.policy.model.eval()
                        logits, _v = self.policy.model(obs_t)

                        logits = mask_invalid_actions(
                            obs_t,
                            logits,
                            enable=self.policy.enable_action_mask,
                        )
                        logits = sanitize_logits_keep_neginf(logits, nan_repl=0.0)

                        probs = F.softmax(logits, dim=-1)

                        top2 = torch.topk(probs, k=2, dim=-1).values
                        p1 = float(top2[0, 0].item())
                        p2 = float(top2[0, 1].item())
                        conf = p1
                        margin = p1 - p2

                        # ✅ Hard guard: si está MUY seguro, no usar BFS
                        if conf > 0.92:
                            use_bfs = False
                        else:
                            # ✅ solo si ambas señales indican duda real
                            if (conf < float(cfg.hybrid_min_conf)) and (
                                margin < float(cfg.hybrid_min_margin)
                            ):
                                use_bfs = True
                                    
                    # C) fallback BFS
                    if use_bfs:
                        path = bfs_plan(self.env.walls, self.env.agent, self.env.goal)
                        if path is not None and len(path) >= 2:
                            a = int(path_to_actions(path)[0])
                            bfs_used += 1

                obs, r, done, info_step = self.env.step(a)
                info = info_step
                steps += 1

            steps_sum += steps
            if info.get("reached", False):
                successes += 1

        if cfg.deterministic:
            self.policy.set_deterministic_seed(None)

        sr = successes / float(cfg.episodes) if cfg.episodes > 0 else 0.0
        avg_steps = steps_sum / float(cfg.episodes) if cfg.episodes > 0 else 0.0
        return {
            "sr": float(sr),
            "avg_steps": float(avg_steps),
            "bfs_rate": float(bfs_used) / float(max(1, steps_sum))
        }