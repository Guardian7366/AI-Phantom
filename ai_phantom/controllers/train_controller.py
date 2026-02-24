# ai_phantom/controllers/train_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from ai_phantom.core.horizon import sync_horizon

import numpy as np
import torch

from ai_phantom.envs.maze.maze_env import MazeConfig
from ai_phantom.envs.maze import MazeEnv
from ai_phantom.agents.ppo import PPOConfig, PPOTrainer, Policy
from ai_phantom.agents.ppo.buffer import RolloutBuffer
from ai_phantom.controllers.eval_controller import EvalController, EvalConfig


@dataclass
class TrainConfig:
    total_updates: int = 500
    phase: int = 0
    seed: int = 42

    eval_every: int = 50
    eval_episodes: int = 100
    deterministic_eval: bool = True

    # Logging
    log_every: int = 1


class TrainController:
    """
    Controlador PPO para 1 env.
    - Rollout + GAE bootstrap correcto.
    - Eval determinista/estocástico.
    """
    def __init__(
        self,
        env: MazeEnv,
        trainer: PPOTrainer,
        policy: Policy,
        buffer: RolloutBuffer,
        cfg: TrainConfig,
        ppo_cfg: PPOConfig,
        device: torch.device,
    ):
        self.env = env
        self.trainer = trainer
        self.policy = policy
        self.buffer = buffer
        self.cfg = cfg
        self.ppo_cfg = ppo_cfg
        self.device = device
        
        sync_horizon([self.env], self.ppo_cfg.rollout_len, name="TrainController")
        self.obs: np.ndarray
        self.info: Dict[str, Any] = {}

        self.episodes = 0
        self.successes = 0

        self._reset_episode()

        # --- Eval env separado (NO tocar self.env) ---
        eval_env_cfg = MazeConfig(**self.env.cfg.__dict__)
        eval_env = MazeEnv(eval_env_cfg, seed=int(self.cfg.seed + 999_999))

        self.evaluator = EvalController(env=eval_env, policy=self.policy, device=self.device)

    def _reset_episode(self) -> None:
        obs, info = self.env.reset(seed=int(self.cfg.seed + self.episodes), phase=int(self.cfg.phase))
        self.obs = obs
        self.info = info

    def train(self) -> None:
        for upd in range(1, int(self.cfg.total_updates) + 1):
            self.buffer.reset()
            rollout_last_done = False

            # ---- Rollout ----
            for _t in range(int(self.ppo_cfg.rollout_len)):
                obs_t = torch.from_numpy(self.obs).unsqueeze(0).to(self.device).float()
                out = self.policy.act(obs_t, deterministic=False)

                action = int(out.action.item())
                logp = float(out.logp.item())
                value = float(out.value.item())

                next_obs, reward, done, info = self.env.step(action)

                self.buffer.add(
                    obs=self.obs,
                    action=action,
                    reward=float(reward),
                    done=bool(done),
                    value=value,
                    logp=logp,
                )

                self.obs = next_obs
                self.info = info
                rollout_last_done = bool(done)

                if done:
                    self.episodes += 1
                    if bool(info.get("reached", False)):
                        self.successes += 1
                    self._reset_episode()

            # ---- Bootstrap para GAE ----
            if rollout_last_done:
                last_value = 0.0
                last_done = True
            else:
                with torch.no_grad():
                    obs_last = torch.from_numpy(self.obs).unsqueeze(0).to(self.device).float()
                    last_value = float(self.policy.value(obs_last).item())
                last_done = False

            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                last_done=last_done,
                gamma=float(self.ppo_cfg.gamma),
                gae_lambda=float(self.ppo_cfg.gae_lambda),
            )

            # ---- Update PPO ----
            metrics = self.trainer.update(self.buffer)

            if metrics.get("nan_abort", 0.0) > 0.5:
                for g in self.trainer.optim.param_groups:
                    g["lr"] = float(g["lr"]) * 0.5
                print("⚠️ nan_abort detected -> lowering LR x0.5")

            if upd % int(self.cfg.log_every) == 0:
                train_sr = (self.successes / self.episodes) if self.episodes > 0 else 0.0
                print(
                    f"[UPD {upd:04d}] episodes={self.episodes:5d} trainSR={train_sr:.3f} "
                    f"pi={metrics['pi_loss']:.4f} vf={metrics['vf_loss']:.4f} ev={metrics['explained_var']:.3f} "
                    f"ent={metrics['entropy']:.4f} kl={metrics['approx_kl']:.5f} stop={int(metrics['early_stop'])} "
                    f"nan={int(metrics.get('nan_abort', 0.0) > 0.5)}"
                )

            # ---- Eval ----
            if upd % int(self.cfg.eval_every) == 0:
                use_walls = bool(getattr(self.env.cfg, "use_walls", False))
                wall_prob = float(getattr(self.env.cfg, "wall_prob", 0.0))

                det = self.evaluator.evaluate(
                    EvalConfig(
                        episodes=int(self.cfg.eval_episodes),
                        phase=int(self.cfg.phase),
                        seed_base=10_000,
                        deterministic=bool(self.cfg.deterministic_eval),
                        rebuild_walls_each_episode=use_walls,
                        walls_seed_base=777 + 90_000,
                        wall_prob=wall_prob if use_walls else None,
                    )
                )
                sto = self.evaluator.evaluate(
                    EvalConfig(
                        episodes=int(self.cfg.eval_episodes),
                        phase=int(self.cfg.phase),
                        seed_base=20_000,
                        deterministic=False,
                    )
                )
                print(
                    f"   EVAL(det): SR={det['sr']:.3f} avg_steps={det['avg_steps']:.1f} | "
                    f"EVAL(sto): SR={sto['sr']:.3f} avg_steps={sto['avg_steps']:.1f}"
                )