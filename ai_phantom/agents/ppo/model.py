from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CnnActorCritic(nn.Module):
    """
    CNN Actor-Critic para obs (C,H,W) como en MazeEnv._make_obs().
    Policy: logits para 4 acciones (UP,DOWN,LEFT,RIGHT)
    Value: V(s)
    """
    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int = 4):
        super().__init__()
        c, h, w = obs_shape
        if num_actions <= 0:
            raise ValueError("num_actions debe ser > 0")

        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.backbone = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # inferir tamaño del flatten
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flat = self.backbone(dummy).flatten(1).shape[1]  # ✅ antes view()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
        )

        self.pi = nn.Linear(256, num_actions)
        self.v = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        # Inicialización razonable para PPO
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Cabezas: policy con gain pequeño para estabilidad
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs: (B,C,H,W) float32
        returns:
          logits: (B, A)
          value: (B,)  (squeeze)
        """
        # ✅ safety net: si llega algo corrupto, no explota el update
        if not torch.isfinite(obs).all():
            obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        x = self.backbone(obs)
        x = self.mlp(x)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value