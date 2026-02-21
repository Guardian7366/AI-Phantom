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
    per_priority_clip: Optional[float] = None  # clip opcional para estabilidad lvl2

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

    # (Opcional) boost automático si hay señales de inestabilidad
    auto_explore_boost_on_td: bool = False
    auto_explore_td_threshold: float = 5.0
    auto_explore_value: float = 0.35
    auto_explore_steps: int = 1500


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
            priority_clip=getattr(self.cfg, "per_priority_clip", None),
        )

        # AMP: solo si CUDA
        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        if use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)

        # Contadores
        self.total_steps = 0  # pasos de entorno (incrementa en remember)
        self.epsilon = float(self.cfg.epsilon_start)

        # Cache gamma
        g = float(self.cfg.gamma)
        self._gamma_t = torch.tensor(g, device=self.device, dtype=torch.float32)
        self._log_gamma = math.log(g) if 0.0 < g < 1.0 else None

        # Epsilon boost temporal
        self._eps_boost_active = False
        self._eps_boost_start_step = 0
        self._eps_boost_end_step = 0
        self._eps_boost_value = 0.0

    # -------------------------
    # Modo explícito (usado por controllers)
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
        t = min(1.0, self.total_steps / float(self.cfg.epsilon_decay_steps))
        eps = float(self.cfg.epsilon_start + t * (self.cfg.epsilon_end - self.cfg.epsilon_start))
        return float(np.clip(eps, 0.0, 1.0))

    def _apply_epsilon_boost(self, eps_sched: float) -> float:
        """
        ✅ FIX CRÍTICO: el boost debe DECER hacia eps_sched.
        Antes: max(..., boost_value) lo dejaba pegado en boost_value.
        """
        if not self._eps_boost_active:
            return eps_sched

        if self.total_steps >= self._eps_boost_end_step:
            self._eps_boost_active = False
            return eps_sched

        span = max(1, self._eps_boost_end_step - self._eps_boost_start_step)
        u = (self.total_steps - self._eps_boost_start_step) / float(span)

        # Interpolación lineal: boost_value -> eps_sched
        eps = float((1.0 - u) * self._eps_boost_value + u * eps_sched)
        # Nunca menos que el schedule
        eps = max(eps, float(eps_sched))
        return float(np.clip(eps, 0.0, 1.0))

    def _update_epsilon(self):
        eps_sched = self._schedule_epsilon()
        self.epsilon = self._apply_epsilon_boost(eps_sched)

    def _start_epsilon_boost(self, *, value: float, steps: int):
        v = float(np.clip(value, 0.0, 1.0))
        s = max(1, int(steps))
        self._eps_boost_active = True
        self._eps_boost_start_step = int(self.total_steps)
        self._eps_boost_end_step = int(self.total_steps + s)
        self._eps_boost_value = v

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
        # ✅ torch.as_tensor evita copias innecesarias cuando obs ya es float32
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if deterministic:
            was_train = self.q.training
            if was_train:
                self.q.eval()
            q = self.q(x)
            a = int(torch.argmax(q, dim=1).item())
            if was_train:
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

    @staticmethod
    def _grad_norm(params) -> float:
        tot = 0.0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            if torch.isfinite(g).all():
                tot += float(g.norm(2).item()) ** 2
        return float(math.sqrt(tot))

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

            with torch.no_grad():
                # Double DQN: acción por online, valor por target
                q_next_online = self.q(next_obs_t)
                next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)

                next_q = self.q_tgt(next_obs_t).gather(1, next_actions)
                gamma_n = self._gamma_pow_n(n_steps_t)

                # dones_t se espera (B,1) float 0/1, igual que rewards_t
                target = rewards_t + (1.0 - dones_t) * gamma_n * next_q

            td_error = (q_values - target).detach()              # (B,1)
            loss_per_item = self.loss_fn(q_values, target)       # (B,1)
            w = weights_t
            if w.dim() == 1:
                w = w.unsqueeze(1)
            loss = (loss_per_item * w).mean()

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

        grad_norm = self._grad_norm(self.q.parameters())

        self.scaler.step(self.optim)
        self.scaler.update()

        # priorities = |td_error|
        td_np = td_error.squeeze(1).detach().abs().cpu().numpy()
        self.buffer.update_priorities(idxs, td_np)

        td_abs_mean = float(np.mean(td_np)) if td_np.size else float("nan")

        self.soft_update()

        # Auto-boost opcional (apagado por default)
        if bool(getattr(self.cfg, "auto_explore_boost_on_td", False)) and np.isfinite(td_abs_mean):
            if td_abs_mean >= float(getattr(self.cfg, "auto_explore_td_threshold", 5.0)):
                self._start_epsilon_boost(
                    value=float(getattr(self.cfg, "auto_explore_value", 0.35)),
                    steps=int(getattr(self.cfg, "auto_explore_steps", 1500)),
                )

        return {
            "loss": float(loss.item()),
            "epsilon": float(self.epsilon),
            "buffer_len": int(len(self.buffer)),
            "beta": float(self.buffer.beta()),
            "td_abs_mean": float(td_abs_mean),
            "q_mean": float(q_values.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
            "grad_norm": float(grad_norm),
        }

    # -------------------------
    # Reproducibilidad / sandbox
    # -------------------------
    def get_state(self) -> Dict[str, Any]:
        st: Dict[str, Any] = {
            "total_steps": int(self.total_steps),
            "epsilon": float(self.epsilon),
            "rng_state": self.rng.bit_generator.state,

            # ✅ extra para reproducibilidad fuerte (sandbox)
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,

            "q_state": {k: v.detach().cpu() for k, v in self.q.state_dict().items()},
            "q_tgt_state": {k: v.detach().cpu() for k, v in self.q_tgt.state_dict().items()},
            "optim_state": self.optim.state_dict(),

            "eps_boost": {
                "active": bool(self._eps_boost_active),
                "start": int(self._eps_boost_start_step),
                "end": int(self._eps_boost_end_step),
                "value": float(self._eps_boost_value),
            },
        }
        if hasattr(self.buffer, "get_state"):
            st["buffer_state"] = self.buffer.get_state()
        return st

    def set_state(self, state: Dict[str, Any]) -> None:
        self.total_steps = int(state["total_steps"])
        self.epsilon = float(state["epsilon"])

        self.rng = np.random.default_rng(0)
        self.rng.bit_generator.state = state["rng_state"]

        # ✅ restaurar RNG torch/cuda si están presentes
        if "torch_rng_state" in state and state["torch_rng_state"] is not None:
            torch.set_rng_state(state["torch_rng_state"])
        if self.device.type == "cuda" and state.get("cuda_rng_state_all", None) is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

        q_sd = {k: v.to(self.device) for k, v in state["q_state"].items()}
        qt_sd = {k: v.to(self.device) for k, v in state["q_tgt_state"].items()}
        self.q.load_state_dict(q_sd)
        self.q_tgt.load_state_dict(qt_sd)

        self.optim.load_state_dict(state["optim_state"])
        # asegurar estados del optimizador en el device correcto
        for st in self.optim.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(self.device)
                    
        eb = state.get("eps_boost", {})
        self._eps_boost_active = bool(eb.get("active", False))
        self._eps_boost_start_step = int(eb.get("start", 0))
        self._eps_boost_end_step = int(eb.get("end", 0))
        self._eps_boost_value = float(eb.get("value", 0.0))

        if "buffer_state" in state and hasattr(self.buffer, "set_state"):
            self.buffer.set_state(state["buffer_state"])