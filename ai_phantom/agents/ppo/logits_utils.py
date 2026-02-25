from __future__ import annotations

import torch


def sanitize_logits_keep_neginf(logits: torch.Tensor, nan_repl: float = 0.0) -> torch.Tensor:
    """
    - Mantiene -inf (masking)
    - Reemplaza NaN/+inf con nan_repl
    - NO rescata filas all -inf aquí (eso se maneja en fix_all_neginf_rows)
    """
    if not torch.is_floating_point(logits):
        logits = logits.float()

    # Fast path: si no hay NaN ni +inf, no toques nada (preserva -inf tal cual)
    if (not torch.isnan(logits).any()) and (not torch.isposinf(logits).any()):
        return logits

    return torch.nan_to_num(
        logits,
        nan=float(nan_repl),
        posinf=float(nan_repl),
        neginf=float("-inf"),
    )