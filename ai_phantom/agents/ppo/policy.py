# ai_phantom/agents/ppo/policy.py
from __future__ import annotations
from .logits_utils import sanitize_logits_keep_neginf
from dataclasses import dataclass
from typing import Optional
from .logits_utils_extra import fix_all_neginf_rows

import torch
from torch.distributions import Categorical

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
    def __init__(
        self,
        model,
        tie_eps: float = 1e-6,
        enable_action_mask: bool = True,
        nan_repl: float = 0.0,
        fallback_action: int = 0,
        ):
        self.model = model
        self.tie_eps = float(tie_eps)
        self.enable_action_mask = bool(enable_action_mask)

        self._det_seed: Optional[int] = None
        self._gen: Optional[torch.Generator] = None
        self.nan_repl = float(nan_repl)
        self.fallback_action = int(fallback_action)

    def set_deterministic_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            self._det_seed = None
            self._gen = None
            return

        seed = int(seed)
        self._det_seed = seed
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool) -> ActOutput:
        self.model.eval()

        logits, value = self.model(obs)
        if value.dim() == 2 and value.size(-1) == 1:
            value = value.squeeze(-1)

        # Action masking (misma lógica que usará el trainer)
        logits = mask_invalid_actions(
            obs,
            logits,
            enable=self.enable_action_mask,
            fallback_action=int(self.fallback_action),
        )

        # ✅ Protección numérica igual que trainer
        logits = sanitize_logits_keep_neginf(logits, nan_repl=float(self.nan_repl))
        logits = fix_all_neginf_rows(logits, fill=0.0, fallback_action=int(self.fallback_action))
        
        row_has_finite = torch.isfinite(logits).any(dim=-1)  # [B]
        if (not bool(row_has_finite.all().item())):
            # Si llegara a pasar (muy raro), reparamos filas malas de forma local
            logits = fix_all_neginf_rows(
                logits, fill=0.0, fallback_action=int(self.fallback_action)
            )

        # ✅ Blindaje extremo: si algo raro dejó logits no finitos, fuerza fallback
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            logits = sanitize_logits_keep_neginf(logits, nan_repl=float(self.nan_repl))
            logits = fix_all_neginf_rows(logits, fill=0.0, fallback_action=int(self.fallback_action))
            
        # Construye distribución EXACTAMENTE como trainer (más estable que probs=exp(log_softmax))
        dist = Categorical(logits=logits)

        if deterministic:
            # Selección determinista con tie-break (sobre logits, equivalente a sobre probs)
            maxlog = logits.max(dim=-1, keepdim=True).values
            is_best = logits >= (maxlog - self.tie_eps)

            actions = []
            for b in range(logits.size(0)):
                best_idx = torch.nonzero(is_best[b], as_tuple=False).squeeze(-1)
                best_idx_cpu = best_idx.detach().to("cpu")

                if best_idx.numel() == 1:
                    a = int(best_idx_cpu.item())
                else:
                    if self._gen is None:
                        # fallback determinista si no hay generador: argmax
                        a = int(torch.argmax(logits[b]).item())
                    else:
                        # tie-break reproducible en CPU
                        w = torch.ones((best_idx.numel(),), dtype=torch.float32)  # CPU
                        j = int(torch.multinomial(w, 1, generator=self._gen).item())
                        a = int(best_idx[j].item())

                actions.append(a)

            action = torch.tensor(actions, device=obs.device, dtype=torch.long)
        else:
            # Modo estocástico: sample directo de Categorical(logits)
            action = dist.sample()

        # logp consistente con dist (evita depender de log_softmax manual)
        chosen_logp = dist.log_prob(action)
        
        # ✅ Solo corregimos NaN/+inf. -inf lo dejamos (si llegara a ocurrir, es señal real de máscara)
        chosen_logp = torch.nan_to_num(chosen_logp, nan=0.0, posinf=0.0)

        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        return ActOutput(action=action, logp=chosen_logp, value=value)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        _, v = self.model(obs)
        if v.dim() == 2 and v.size(-1) == 1:
            v = v.squeeze(-1)
        return v