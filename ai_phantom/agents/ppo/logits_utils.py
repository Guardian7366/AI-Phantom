# ai_phantom/agents/ppo/logits_utils.py
from __future__ import annotations
import torch

def sanitize_logits_keep_neginf(logits: torch.Tensor, nan_repl: float = 0.0) -> torch.Tensor:
    """
    - Mantiene -inf (masking)
    - Reemplaza NaN/+inf con nan_repl
    - Si una fila queda todo -inf, la rescata a ceros
    """
    logits = torch.nan_to_num(
        logits,
        nan=float(nan_repl),
        posinf=float(nan_repl),
        neginf=float("-inf"),
    )
    all_neginf = torch.isneginf(logits).all(dim=-1)
    if bool(all_neginf.any()):
        logits = logits.clone()
        logits[all_neginf] = 0.0
    return logits