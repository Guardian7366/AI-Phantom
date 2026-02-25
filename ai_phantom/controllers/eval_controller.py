# ai_phantom/controllers/eval_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from ai_phantom.agents.ppo.logits_utils_extra import fix_all_neginf_rows
from ai_phantom.agents.ppo.action_mask import mask_invalid_actions
from ai_phantom.agents.ppo.logits_utils import sanitize_logits_keep_neginf
from ai_phantom.planners.bfs import bfs_plan, path_to_actions


@dataclass
class EvalConfig:
    episodes: int
    phase: int

    seed_base: int = 10_000
    deterministic: bool = True

    rebuild_walls_each_episode: bool = False
    walls_seed_base: int = 777
    wall_prob: Optional[float] = None

    # ✅ Hybrid controller (PPO + BFS fallback)
    hybrid: bool = False
    hybrid_on_loop: bool = True

    # usar BFS si la policy está incierta
    hybrid_min_conf: float = 0.55
    hybrid_min_margin: float = 0.08  # diferencia top1-top2

    # hard guard: si la policy está MUY segura, no usar BFS
    hybrid_hard_conf: float = 0.92

    # señal inmediata del env (evita lag de 1 paso)
    hybrid_use_env_signals: bool = True


class EvalController:
    def __init__(self, env, policy, device: torch.device):
        self.env = env
        self.policy = policy
        self.device = device

    @torch.no_grad()
    
    def evaluate(self, cfg: EvalConfig) -> Dict[str, Any]:
        # ✅ Evaluación siempre en modo eval (sin dropout / BN updates)
        self.policy.model.eval()

        successes = 0
        steps_sum = 0
        bfs_used_steps = 0
        bfs_try_steps = 0          # ✅ NUEVO: veces que quisimos usar BFS
        bfs_fail_steps = 0         # ✅ NUEVO: veces que BFS no pudo dar acción
                

        for ep in range(int(cfg.episodes)):
            seed = int(cfg.seed_base + ep)

            # ✅ rebuild walls por episodio (multi-walls) si el env lo soporta
            if bool(cfg.rebuild_walls_each_episode) and hasattr(self.env, "rebuild_walls"):
                ws = int(cfg.walls_seed_base + ep)
                self.env.rebuild_walls(seed=ws, wall_prob=cfg.wall_prob)

            if bool(cfg.deterministic):
                self.policy.set_deterministic_seed(seed)

            obs, info = self.env.reset(seed=seed, phase=int(cfg.phase))
            info = dict(info) if isinstance(info, dict) else {}

            done = False
            steps = 0
            bfs_fail_ep = 0
            bfs_disable_ep = False

            while not done:
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device).float()

                # 1) acción PPO
                out = self.policy.act(obs_t, deterministic=bool(cfg.deterministic))
                a = int(out.action.item())

                if bool(cfg.hybrid):
                    use_bfs = False

                    if bfs_disable_ep:
                        use_bfs = False
                        
                    info_looped = bool(info.get("looped", False))
                    no_prog = int(info.get("no_progress", 0))
                    loop_hits = int(info.get("loop_hits", 0))

                    if bool(cfg.hybrid_on_loop) and info_looped:
                        use_bfs = True

                    if (not use_bfs) and bool(cfg.hybrid_use_env_signals):
                        stag_steps = int(getattr(self.env.cfg, "stagnation_steps", 999999))
                        if (loop_hits >= 2) or (no_prog >= stag_steps):
                            use_bfs = True

                    # B) baja confianza + bajo margen
                    if not use_bfs:
                        logits, _v = self.policy.model(obs_t)

                        logits = mask_invalid_actions(
                            obs_t,
                            logits,
                            enable=bool(getattr(self.policy, "enable_action_mask", True)),
                        )

                        logits = sanitize_logits_keep_neginf(logits, nan_repl=0.0)
                        logits = fix_all_neginf_rows(logits, fill=0.0)

                        logp_all = F.log_softmax(logits, dim=-1)
                        probs = logp_all.exp()

                        if not torch.isfinite(probs).all():
                            # fallback: no uses BFS por "confianza", conf alta para bloquear
                            use_bfs = False
                            probs = torch.zeros_like(probs)
                            probs[0, a] = 1.0

                        top2 = torch.topk(probs, k=2, dim=-1).values
                        p1 = float(top2[0, 0].item())
                        p2 = float(top2[0, 1].item())
                        conf = p1
                        margin = p1 - p2

                        # ✅ Hard guard: si está MUY seguro, no usar BFS
                        if conf > float(cfg.hybrid_hard_conf):
                            use_bfs = False
                        else:
                            if (conf < float(cfg.hybrid_min_conf)) and (
                                margin < float(cfg.hybrid_min_margin)
                            ):
                                use_bfs = True

                    # C) fallback BFS (✅ safe + telemetry)
                    if use_bfs:
                        bfs_try_steps += 1
                        path = bfs_plan(self.env.walls, self.env.agent, self.env.goal)

                        if path is not None and len(path) >= 2:
                            a = int(path_to_actions(path)[0])
                            bfs_used_steps += 1
                        else:
                            # BFS quiso usarse pero falló (no path / path degenerado)
                            bfs_fail_steps += 1
                            bfs_fail_ep += 1
                            if bfs_fail_ep >= 3:
                                bfs_disable_ep = True

                            # ✅ optional: “safe action” si PPO diera algo raro (muy raro, pero gratis)
                            # Elegimos la primera acción que no choque; si todas chocan, deja 'a' como estaba.
                            try:
                                ar, ac = self.env.agent
                                for aa, (dr, dc) in self.env.ACTIONS.items():
                                    nr, nc = ar + dr, ac + dc
                                    if 0 <= nr < self.env.cfg.height and 0 <= nc < self.env.cfg.width and (not self.env.walls[nr, nc]):
                                        a = int(aa)
                                        break
                            except Exception:
                                pass

                obs, _r, done, info_step = self.env.step(a)
                info = info_step
                steps += 1

            steps_sum += steps
            if bool(info.get("reached", False)):
                successes += 1

        if bool(cfg.deterministic):
            self.policy.set_deterministic_seed(None)

        sr = successes / float(cfg.episodes) if cfg.episodes > 0 else 0.0
        avg_steps = steps_sum / float(cfg.episodes) if cfg.episodes > 0 else 0.0

        return {
            "sr": float(sr),
            "avg_steps": float(avg_steps),
            "bfs_rate": float(bfs_used_steps) / float(max(1, steps_sum)),
            "bfs_used_steps": int(bfs_used_steps),
            "bfs_try_steps": int(bfs_try_steps),       # ✅ NUEVO
            "bfs_fail_steps": int(bfs_fail_steps),     # ✅ NUEVO
            "steps_sum": int(steps_sum),
            "bfs_try_rate": float(bfs_try_steps) / float(max(1, steps_sum)),
        }