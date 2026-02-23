# ai_phantom/agents/ppo/action_mask.py
from __future__ import annotations

import torch


def mask_invalid_actions(
    obs: torch.Tensor,
    logits: torch.Tensor,
    enable: bool = True,
) -> torch.Tensor:
    """
    Aplica action-masking para evitar acciones inválidas (chocar con paredes / salir bounds)
    usando canales de la observación.

    Espera obs con shape [B,C,H,W] y canales:
      0: walls (1.0=pared)
      1: agent (one-hot)

    logits: [B,A] con A=4 (UP,DOWN,LEFT,RIGHT)

    Retorna logits con inválidas en -inf. Si por alguna razón no hay ninguna válida
    en un batch, no mask (evita NaNs).
    """
    if (not enable) or (obs.dim() != 4) or (logits.dim() != 2):
        return logits

    B, C, H, W = obs.shape
    if C < 2 or logits.size(-1) != 4:
        return logits

    walls = obs[:, 0]  # [B,H,W]
    agent = obs[:, 1]  # [B,H,W]

    flat_idx = agent.view(B, -1).argmax(dim=1)  # [B]
    ar = (flat_idx // W).to(torch.long)
    ac = (flat_idx % W).to(torch.long)

    # 0=UP,1=DOWN,2=LEFT,3=RIGHT
    nr = torch.stack([ar - 1, ar + 1, ar, ar], dim=1)      # [B,4]
    nc = torch.stack([ac, ac, ac - 1, ac + 1], dim=1)      # [B,4]

    inb = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)      # [B,4]

    nr_clamped = nr.clamp(0, H - 1)
    nc_clamped = nc.clamp(0, W - 1)

    b_idx = torch.arange(B, device=obs.device).unsqueeze(1).expand(B, 4)  # [B,4]
    is_wall = walls[b_idx, nr_clamped, nc_clamped] > 0.5                  # [B,4]

    valid = inb & (~is_wall)  # [B,4]

    any_valid = valid.any(dim=1, keepdim=True)  # [B,1]
    valid = torch.where(any_valid, valid, torch.ones_like(valid, dtype=torch.bool))

    return logits.masked_fill(~valid, float("-inf"))