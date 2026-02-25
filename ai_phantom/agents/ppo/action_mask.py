# ai_phantom/agents/ppo/action_mask.py
from __future__ import annotations
import torch


def mask_invalid_actions(
    obs: torch.Tensor,
    logits: torch.Tensor,
    enable: bool = True,
    fallback_action: int = 0,
) -> torch.Tensor:
    if (not enable) or (obs.dim() != 4) or (logits.dim() != 2):
        return logits

    B, C, H, W = obs.shape
    if C < 2 or logits.size(-1) != 4:
        return logits
    
    fa = int(fallback_action)
    if fa < 0:
        fa = 0
    if fa >= logits.size(-1):
        fa = logits.size(-1) - 1

    walls = obs[:, 0]   # [B,H,W]
    agent = obs[:, 1]   # [B,H,W]

    with torch.no_grad():
        agent_sum = agent.flatten(1).sum(dim=1)   # [B]
        good = agent_sum > 0.5                    # [B]
        if not good.any().item():
            valid = torch.zeros((B, 4), device=obs.device, dtype=torch.bool)
            valid[:, fa] = True
            return logits.masked_fill(~valid, float("-inf"))
        agent = torch.where(good.view(B, 1, 1), agent, torch.zeros_like(agent))

        flat_idx = agent.flatten(1).argmax(dim=1)  # [B]
        ar = (flat_idx // W).to(torch.long)
        ac = (flat_idx % W).to(torch.long)

        nr = torch.stack([ar - 1, ar + 1, ar, ar], dim=1)  # [B,4]
        nc = torch.stack([ac, ac, ac - 1, ac + 1], dim=1)  # [B,4]

        inb = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)  # [B,4]

        nr_clamped = nr.clamp(0, H - 1)
        nc_clamped = nc.clamp(0, W - 1)

        b_idx = torch.arange(B, device=obs.device).unsqueeze(1).expand(B, 4)  # [B,4]
        is_wall = walls[b_idx, nr_clamped, nc_clamped] > 0.5                  # [B,4]
        valid = inb & (~is_wall)                                              # [B,4]

        # ✅ si una fila es "bad" (agent channel corrupto), no la mask (dejamos logits intactos)
        valid = torch.where(good.unsqueeze(1), valid, torch.ones_like(valid))

        # ✅ Si una fila "good" queda sin válidas, habilita SOLO 1 acción (fallback determinista)
        any_valid = valid.any(dim=1)  # [B]
        bad_good = good & (~any_valid)

        if bad_good.any().item():
            valid = valid.clone()
            valid[bad_good] = False
            valid[bad_good, fa] = True

    valid = valid.to(torch.bool)
    if bool(valid.all().item()):
        return logits
    return logits.masked_fill(~valid, float("-inf"))