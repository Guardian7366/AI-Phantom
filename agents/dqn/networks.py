# agents/dqn/networks.py
from __future__ import annotations

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    """
    Entrada: (B, C=3, H, W)  (H=W=8 en tu Maze)
    Salida:  (B, A) Q-values

    FIX CRÍTICO:
    - Se elimina Global Average Pooling (AdaptiveAvgPool2d(1)) porque destruye la info espacial.
    - Se mantiene info espacial con downsampling controlado (stride=2) y luego flatten.
    """

    def __init__(self, in_channels: int = 3, num_actions: int = 4):
        super().__init__()
        self.num_actions = int(num_actions)

        def _safe_groups(cout: int, preferred: int = 8) -> int:
            g = min(int(preferred), int(cout))
            while g > 1 and (cout % g) != 0:
                g -= 1
            return max(1, g)

        def block(cin: int, cout: int, *, stride: int = 1) -> nn.Sequential:
            g = _safe_groups(cout, preferred=8)
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, stride=int(stride), padding=1, bias=False),
                nn.GroupNorm(num_groups=g, num_channels=cout),
                nn.ReLU(inplace=True),
            )

        # Para H=W=8:
        # - conv stride1 -> 8x8
        # - conv stride2 -> 4x4
        # Mantiene suficiente info espacial sin explotar parámetros.
        self.features = nn.Sequential(
            block(in_channels, 64, stride=1),   # 8x8
            block(64, 64, stride=1),            # 8x8
            block(64, 128, stride=2),           # 4x4
            block(128, 128, stride=1),          # 4x4
            nn.Flatten(),                       # (B, 128*4*4) = (B, 2048)
        )

        feat_dim = 128 * 4 * 4  # 2048 para grid 8x8 con un stride2

        # Heads dueling
        hidden = 512
        self.value = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                # kaiming para capas con ReLU antes; para las últimas también es ok (bias=0)
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # (Opcional defensivo) pequeña inicialización para estabilizar el dueling al inicio:
        # hace que advantage arranque cerca de 0
        last_adv = self.advantage[-1]
        if isinstance(last_adv, nn.Linear):
            nn.init.constant_(last_adv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        z = self.features(x)                  # (B, feat_dim)
        v = self.value(z)                     # (B,1)
        a = self.advantage(z)                 # (B,A)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q