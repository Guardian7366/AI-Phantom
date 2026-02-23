# ai_phantom/agents/ppo/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .action_mask import mask_invalid_actions


@dataclass
class ActOutput:
    action: torch.Tensor  # shape [B]
    logp: torch.Tensor    # shape [B]
    value: torch.Tensor   # shape [B]


class Policy:
    """
    Wrapper de inferencia sobre el Actor-Critic.

    - deterministic=True hace tie-break determinista (por seed) cuando hay empate (o casi empate).
    - action masking para evitar acciones inválidas (choques/bounds) usando obs channels.
    """
    def __init__(self, model, tie_eps: float = 1e-6, enable_action_mask: bool = True):
        self.model = model
        self.tie_eps = float(tie_eps)
        self.enable_action_mask = bool(enable_action_mask)

        self._det_seed: Optional[int] = None
        self._gen: Optional[torch.Generator] = None

    def set_deterministic_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            self._det_seed = None
            self._gen = None
            return

        seed = int(seed)
        self._det_seed = seed
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)

    @staticmethod
    def _sanitize_logits_like_trainer(logits: torch.Tensor, nan_repl: float = 0.0) -> torch.Tensor:
        """
        Mantiene -inf (del masking). Solo corrige NaN/+inf.
        Si una fila queda completamente inválida (todo -inf), la rescata con ceros.
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

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool) -> ActOutput:
        self.model.eval()

        logits, value = self.model(obs)
        if value.dim() == 2 and value.size(-1) == 1:
            value = value.squeeze(-1)

        # Action masking (misma lógica que usará el trainer)
        logits = mask_invalid_actions(obs, logits, enable=self.enable_action_mask)

        # ✅ Protección numérica igual que trainer
        logits = self._sanitize_logits_like_trainer(logits, nan_repl=0.0)

        logp_all = F.log_softmax(logits, dim=-1)
        probs = logp_all.exp()

        if deterministic:
            maxp = probs.max(dim=-1, keepdim=True).values
            is_best = probs >= (maxp - self.tie_eps)

            actions = []
            for b in range(probs.size(0)):
                best_idx = torch.nonzero(is_best[b], as_tuple=False).squeeze(-1)
                if best_idx.numel() == 1:
                    a = int(best_idx.item())
                else:
                    if self._gen is None:
                        a = int(torch.argmax(probs[b]).item())
                    else:
                        w = torch.ones((best_idx.numel(),), dtype=torch.float32)  # CPU
                        j = int(torch.multinomial(w, 1, generator=self._gen).item())
                        a = int(best_idx[j].item())
                actions.append(a)

            action = torch.tensor(actions, device=obs.device, dtype=torch.long)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        chosen_logp = logp_all.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        return ActOutput(action=action, logp=chosen_logp, value=value)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        _, v = self.model(obs)
        if v.dim() == 2 and v.size(-1) == 1:
            v = v.squeeze(-1)
        return v