# agents/ppo/networks.py
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvBackbone(nn.Module):
    """
    CNN simple + AdaptiveAvgPool2d => size-agnostic.
    Entrada: (B, C=3, H, W)
    Salida: (B, feat_dim)
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B,128)
        x = self.fc(x)                            # (B,feat_dim)
        return x


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic:
    - Actor: logits sobre acciones
    - Critic: V(s)
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = 4,
        feat_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone = ConvBackbone(in_channels=in_channels, feat_dim=feat_dim)

        self.actor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        logits = self.actor(feat)   # (B,A)
        value = self.critic(feat)   # (B,1)
        return logits, value