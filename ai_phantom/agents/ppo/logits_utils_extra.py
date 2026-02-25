from __future__ import annotations

import torch


def fix_all_neginf_rows(
    logits: torch.Tensor,
    fill: float = 0.0,
    fallback_action: int = 0,
) -> torch.Tensor:
    """
    Rescata filas degeneradas para evitar NaNs en softmax/Categorical.

    Solo toca filas donde NO existe ninguna acción válida numéricamente:
      - todas -inf
      - o ninguna entrada finita (todo NaN / +/-inf)

    En esas filas:
      - pone todo a -inf
      - y habilita SOLO una acción fallback con 'fill' (0.0 => prob ~1.0 para esa acción tras softmax)
    """
    if not float(fill) == float(fill):  # NaN check simple
        fill = 0.0

    # Esperamos [B, A]. Si no, no tocamos nada.
    if logits.dim() != 2:
        return logits

    # Edge case: A == 0 (no hay acciones) -> no hay nada que rescatar
    if logits.size(-1) == 0:
        return logits

    # Filas totalmente degeneradas:
    # - all_neginf: todas son -inf
    # - no_finite: ninguna es finita (NaN / +/-inf)
    all_neginf = torch.isneginf(logits).all(dim=-1)
    any_finite = torch.isfinite(logits).any(dim=-1)
    bad = all_neginf | (~any_finite)

    # En CUDA, usar `.item()` es claro y evita ambigüedad del `if tensor:`
    if bad.any().item():
        out = logits.clone()
        out[bad] = float("-inf")

        fa = int(fallback_action)
        if fa < 0:
            fa = 0
        if fa >= out.size(-1):
            fa = out.size(-1) - 1

        out[bad, fa] = float(fill)
        return out

    return logits