# agents/dqn/networks.py
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block ligero (Conv-GN-ReLU-Conv-GN) + skip.
    - Muy estable en RL (batch pequeños) con GroupNorm.
    """
    def __init__(self, channels: int, *, groups_preferred: int = 8):
        super().__init__()
        c = int(channels)

        def _safe_groups(cout: int, preferred: int = 8) -> int:
            g = min(int(preferred), int(cout))
            while g > 1 and (cout % g) != 0:
                g -= 1
            return max(1, g)

        g = _safe_groups(c, preferred=groups_preferred)

        self.conv1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=g, num_channels=c)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=g, num_channels=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.gn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.gn2(y)
        y = y + x
        y = self.act(y)
        return y


class DuelingQNetwork(nn.Module):
    """
    Size-agnostic Dueling Q-Network.

    Entrada: (B, C=3, H, W)  con H,W variables
    Salida:  (B, A) Q-values

    Claves:
    - Preserva estructura espacial con convs + residual.
    - Downsampling controlado (stride=2).
    - Adaptive pooling fija el tamaño de features => MLP fijo => size-agnostic.
    - GroupNorm: estable en RL con batch pequeños.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_actions: int = 4,
        *,
        pool_hw: int = 4,          # tamaño fijo tras pooling (4x4 suele ir muy bien)
        hidden: int = 512,
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.in_channels = int(in_channels)
        self.pool_hw = int(pool_hw)
        self.hidden = int(hidden)

        def _safe_groups(cout: int, preferred: int = 8) -> int:
            g = min(int(preferred), int(cout))
            while g > 1 and (cout % g) != 0:
                g -= 1
            return max(1, g)

        def conv_gn_relu(cin: int, cout: int, *, stride: int = 1) -> nn.Sequential:
            g = _safe_groups(cout, preferred=8)
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=int(stride), padding=1, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=cout),
                nn.ReLU(inplace=True),
            )

        # Backbone (agnóstico al tamaño)
        self.stem = conv_gn_relu(self.in_channels, 64, stride=1)
        self.res1 = ResidualBlock(64)

        self.down = conv_gn_relu(64, 128, stride=2)
        self.res2 = ResidualBlock(128)

        # ✅ FIX size-agnostic: fija tamaño espacial
        # Nota: AvgPool funciona bien y es estable; MaxPool también valdría pero suele ser más ruidoso en RL.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_hw, self.pool_hw))
        self.flatten = nn.Flatten()

        # feat_dim fijo por construcción: 128 * pool_hw * pool_hw
        feat_dim = 128 * self.pool_hw * self.pool_hw

        self.value = nn.Sequential(
            nn.Linear(feat_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(feat_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Estabiliza dueling: advantage arranca cerca de 0
        last_adv = self.advantage[-1]
        if isinstance(last_adv, nn.Linear):
            if last_adv.bias is not None:
                nn.init.constant_(last_adv.bias, 0.0)
            nn.init.uniform_(last_adv.weight, a=-1e-3, b=1e-3)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compat AMP: respeta float16/bfloat16 si viene en autocast
        if not torch.is_floating_point(x):
            x = x.float()

        z = self._forward_features(x)     # (B, feat_dim)
        v = self.value(z)                 # (B,1)
        a = self.advantage(z)             # (B,A)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q