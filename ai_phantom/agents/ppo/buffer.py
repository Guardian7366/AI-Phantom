from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logp_old: torch.Tensor
    values_old: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    is_teacher: torch.Tensor


class RolloutBuffer:
    """
    Almacena una trayectoria de longitud T para PPO y calcula GAE.
    - 1 env (por ahora)
    - Optimizado para transfer CPU->GPU:
        * buffers en CPU con pin_memory (si device=cuda)
        * iter_minibatches usa non_blocking=True con pin_memory real
    """
    def __init__(
        self,
        rollout_len: int,
        obs_shape: Tuple[int, int, int],
        device: torch.device,
        pin_memory: Optional[bool] = None,
    ):
        self.T = int(rollout_len)
        self.obs_shape = obs_shape
        self.device = device

        if pin_memory is None:
            pin_memory = (device.type == "cuda")
        self.pin_memory = bool(pin_memory)

        c, h, w = obs_shape

        # Buffers en CPU (pinned si aplica)
        self.obs = torch.zeros((self.T, c, h, w), dtype=torch.float32, pin_memory=self.pin_memory)
        self.actions = torch.zeros((self.T,), dtype=torch.int64, pin_memory=self.pin_memory)
        self.rewards = torch.zeros((self.T,), dtype=torch.float32, pin_memory=self.pin_memory)
        self.dones = torch.zeros((self.T,), dtype=torch.float32, pin_memory=self.pin_memory)
        self.values = torch.zeros((self.T,), dtype=torch.float32, pin_memory=self.pin_memory)
        self.logps = torch.zeros((self.T,), dtype=torch.float32, pin_memory=self.pin_memory)
        self.is_teacher = torch.zeros((self.T,), dtype=torch.float32, pin_memory=self.pin_memory)
        self.advantages = torch.zeros((self.T,), dtype=torch.float32, pin_memory=self.pin_memory)
        self.returns = torch.zeros((self.T,), dtype=torch.float32, pin_memory=self.pin_memory)

        self.ptr = 0
        self.full = False

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        logp: float,
        is_teacher: bool = False,
    ) -> None:
        if self.ptr >= self.T:
            raise RuntimeError("RolloutBuffer está lleno. Llama reset() o compute_returns_and_advantages().")

        # Asegura float32 y contiguo (barato y evita sorpresas)
        if obs.dtype != np.float32:
            obs = obs.astype(np.float32, copy=False)
        if not obs.flags["C_CONTIGUOUS"]:
            obs = np.ascontiguousarray(obs)

        # PRO: from_numpy evita copias extra y es más consistente aquí
        self.obs[self.ptr].copy_(torch.from_numpy(obs))
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = 1.0 if bool(done) else 0.0
        self.values[self.ptr] = float(value)
        self.logps[self.ptr] = float(logp)
        self.is_teacher[self.ptr] = 1.0 if bool(is_teacher) else 0.0

        self.ptr += 1
        if self.ptr == self.T:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float, last_done: bool, gamma: float, gae_lambda: float) -> None:
        if not self.full:
            raise RuntimeError("Buffer no está lleno: junta T pasos antes de compute_returns_and_advantages().")

        gamma = float(gamma)
        lam = float(gae_lambda)

        last_gae = 0.0
        # Si el rollout terminó en terminal, no bootstrapees
        next_nonterminal_last = 0.0 if bool(last_done) else 1.0
        next_value_last = float(last_value)

        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_value = next_value_last
                next_nonterminal = next_nonterminal_last
            else:
                next_value = float(self.values[t + 1].item())
                # ✅ done almacenado es done_t (resultado del paso t), así que el gate es (1 - done_t)
                next_nonterminal = 1.0 - float(self.dones[t].item())

            delta = float(self.rewards[t].item()) + gamma * next_value * next_nonterminal - float(self.values[t].item())
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            self.advantages[t] = last_gae

        # ✅ NO reasignes tensores (ver punto 2)
        self.returns.copy_(self.advantages + self.values)

        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std(unbiased=False).clamp_min(1e-8)
        self.advantages.sub_(adv_mean).div_(adv_std)

    def iter_minibatches(self, minibatch_size: int, shuffle: bool = True) -> Iterator[RolloutBatch]:
        if not self.full:
            raise RuntimeError("Buffer no lleno.")
        mb = int(minibatch_size)
        if mb <= 0 or mb > self.T:
            raise ValueError(f"minibatch_size inválido: {mb}")

        idx = np.arange(self.T)
        if shuffle:
            np.random.shuffle(idx)

        for start in range(0, self.T, mb):
            end = min(start + mb, self.T)
            b = idx[start:end]

            # non_blocking funciona “de verdad” si pin_memory=True
            yield RolloutBatch(
                obs=self.obs[b].to(self.device, non_blocking=True),
                actions=self.actions[b].to(self.device, non_blocking=True),
                logp_old=self.logps[b].to(self.device, non_blocking=True),
                values_old=self.values[b].to(self.device, non_blocking=True),
                returns=self.returns[b].to(self.device, non_blocking=True),
                advantages=self.advantages[b].to(self.device, non_blocking=True),
                is_teacher=self.is_teacher[b].to(self.device, non_blocking=True),
            )