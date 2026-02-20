from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.networks import DuelingQNetwork
from agents.dqn.replay_buffer import PrioritizedReplayBuffer


@dataclass
class DQNConfig:
    num_actions: int = 4
    gamma: float = 0.99
    lr: float = 2.5e-4
    batch_size: int = 64
    replay_size: int = 60_000

    # Double DQN + target soft update
    tau: float = 0.005

    # PER + n-step
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 200_000
    n_step: int = 5

    # Epsilon-greedy
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 150_000

    # Entrenamiento
    warmup_steps: int = 2_000
    update_every: int = 1
    grad_clip_norm: float = 10.0

    # AMP (fp16)
    use_amp: bool = True

    # seed para PER sampling + RNG del agente
    seed: int = 0


class DQNAgent:
    def __init__(self, cfg: Optional[DQNConfig] = None, device: Optional[torch.device] = None):
        self.cfg = cfg or DQNConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RNG propio (reproducible) + seeds torch
        seed = int(getattr(self.cfg, "seed", 0))
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.q = DuelingQNetwork(in_channels=3, num_actions=self.cfg.num_actions).to(self.device)
        self.q_tgt = DuelingQNetwork(in_channels=3, num_actions=self.cfg.num_actions).to(self.device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()  # target siempre en eval

        self.optim = optim.Adam(self.q.parameters(), lr=float(self.cfg.lr))
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.buffer = PrioritizedReplayBuffer(
            capacity=int(self.cfg.replay_size),
            alpha=float(self.cfg.per_alpha),
            beta_start=float(self.cfg.per_beta_start),
            beta_frames=int(self.cfg.per_beta_frames),
            n_step=int(self.cfg.n_step),
            gamma=float(self.cfg.gamma),
            seed=seed,
        )

        # AMP: solo si CUDA
        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        if use_amp:
            # torch.amp.GradScaler(device="cuda", enabled=True) (PyTorch >= 2)
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            # No-op scaler (más compatible que pasar "cuda" cuando estás en cpu)
            self.scaler = torch.amp.GradScaler(enabled=False)

        # Contadores
        self.total_steps = 0  # pasos de entorno (incrementa en remember)
        self.epsilon = float(self.cfg.epsilon_start)

        # Cache gamma como tensor
        g = float(self.cfg.gamma)
        self._gamma_t = torch.tensor(g, device=self.device, dtype=torch.float32)
        self._log_gamma = math.log(g) if 0.0 < g < 1.0 else None

        # Epsilon boost temporal
        self._eps_boost_active = False
        self._eps_boost_start_step = 0
        self._eps_boost_end_step = 0
        self._eps_boost_value = 0.0

    # -------------------------
    # Modo (opcional) explícito
    # -------------------------
    def train_mode(self):
        self.q.train()

    def eval_mode(self):
        self.q.eval()
        self.q_tgt.eval()

    # -------------------------
    # Epsilon schedule
    # -------------------------
    def _schedule_epsilon(self) -> float:
        # lineal de start -> end
        t = min(1.0, self.total_steps / float(self.cfg.epsilon_decay_steps))
        eps = float(self.cfg.epsilon_start + t * (self.cfg.epsilon_end - self.cfg.epsilon_start))
        return float(np.clip(eps, 0.0, 1.0))

    def _apply_epsilon_boost(self, eps_sched: float) -> float:
        if not self._eps_boost_active:
            return eps_sched

        if self.total_steps >= self._eps_boost_end_step:
            self._eps_boost_active = False
            return eps_sched

        span = max(1, self._eps_boost_end_step - self._eps_boost_start_step)
        u = (self.total_steps - self._eps_boost_start_step) / float(span)

        eps_boost = float((1.0 - u) * self._eps_boost_value + u * eps_sched)
        eps = max(eps_boost, eps_sched, float(self._eps_boost_value))
        return float(np.clip(eps, 0.0, 1.0))

    def _update_epsilon(self):
        eps_sched = self._schedule_epsilon()
        self.epsilon = self._apply_epsilon_boost(eps_sched)

    def _start_epsilon_boost(self, *, value: float, steps: int):
        self._eps_boost_active = True
        self._eps_boost_start_step = int(self.total_steps)
        self._eps_boost_end_step = int(self.total_steps + max(1, int(steps)))
        self._eps_boost_value = float(value)

    def on_curriculum_advanced(
        self,
        *,
        old_level: int,
        new_level: int,
        epsilon_reset_on_advance: bool,
        epsilon_reset_value: float,
        epsilon_reset_steps: int,
        reset_replay_on_advance: bool,
    ):
        if epsilon_reset_on_advance:
            self._start_epsilon_boost(value=float(epsilon_reset_value), steps=int(epsilon_reset_steps))

        if reset_replay_on_advance and hasattr(self.buffer, "reset"):
            self.buffer.reset()

    def trigger_exploration_boost(self, *, value: float, steps: int):
        self._start_epsilon_boost(value=float(value), steps=int(steps))

    # -------------------------
    # Acción
    # -------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        x = torch.tensor(obs[None, ...], dtype=torch.float32, device=self.device)

        if deterministic:
            # ✅ blindaje: si la red tiene dropout/BN, esto hace greedy consistente
            was_training = self.q.training
            if was_training:
                self.q.eval()
            q = self.q(x)
            a = int(torch.argmax(q, dim=1).item())
            if was_training:
                self.q.train()
            return a

        self._update_epsilon()

        if float(self.rng.random()) < float(self.epsilon):
            return int(self.rng.integers(0, int(self.cfg.num_actions)))

        q = self.q(x)
        return int(torch.argmax(q, dim=1).item())

    # -------------------------
    # Memoria / aprendizaje
    # -------------------------
    def remember(self, obs, action, reward, next_obs, done):
        # ✅ total_steps = pasos de entorno (alineado con epsilon schedule)
        self.total_steps += 1
        self.buffer.push(obs, action, reward, next_obs, done)

    def soft_update(self):
        tau = float(self.cfg.tau)
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
                pt.data.mul_(1.0 - tau).add_(tau * p.data)

    def _gamma_pow_n(self, n_steps_t: torch.Tensor) -> torch.Tensor:
        """
        n_steps_t: (B,1) int64
        retorna gamma ** n (B,1) float32
        """
        if self._log_gamma is not None:
            return torch.exp(n_steps_t.float() * float(self._log_gamma))
        return torch.pow(self._gamma_t, n_steps_t.float())

    def learn(self) -> Dict[str, Any]:
        # Warmup
        if len(self.buffer) < int(self.cfg.warmup_steps):
            return {"loss": None, "epsilon": float(self.epsilon), "buffer_len": int(len(self.buffer))}

        # Update frequency ligada a pasos de entorno
        ue = int(self.cfg.update_every)
        if ue > 1 and (self.total_steps % ue != 0):
            return {"loss": None, "epsilon": float(self.epsilon), "buffer_len": int(len(self.buffer))}

        obs_t, actions_t, rewards_t, next_obs_t, dones_t, n_steps_t, idxs, weights_t = self.buffer.sample(
            int(self.cfg.batch_size), self.device
        )

        autocast_enabled = bool(self.scaler.is_enabled())
        with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
            q_values = self.q(obs_t).gather(1, actions_t)

            # Target Double DQN sin gradiente
            with torch.no_grad():
                next_actions = torch.argmax(self.q(next_obs_t), dim=1, keepdim=True)
                next_q = self.q_tgt(next_obs_t).gather(1, next_actions)

                gamma_n = self._gamma_pow_n(n_steps_t)
                target = rewards_t + (1.0 - dones_t) * gamma_n * next_q

            td_error = (q_values - target).detach()  # (B,1)
            loss_per_item = self.loss_fn(q_values, target)  # (B,1)
            loss = (loss_per_item * weights_t.unsqueeze(1)).mean()

            if not torch.isfinite(loss):
                self.optim.zero_grad(set_to_none=True)
                return {
                    "loss": None,
                    "epsilon": float(self.epsilon),
                    "buffer_len": int(len(self.buffer)),
                    "skipped": "non_finite_loss",
                }

        self.optim.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), float(self.cfg.grad_clip_norm))

        self.scaler.step(self.optim)
        self.scaler.update()

        # priorities = |td_error|
        self.buffer.update_priorities(idxs, td_error.squeeze(1).cpu().numpy())
        self.soft_update()

        return {
            "loss": float(loss.item()),
            "epsilon": float(self.epsilon),
            "buffer_len": int(len(self.buffer)),
            "beta": float(self.buffer.beta()),
        }