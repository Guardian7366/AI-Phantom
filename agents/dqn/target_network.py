# agents/dqn/target_network.py
from __future__ import annotations

import torch
import torch.nn as nn


class TargetNetwork:
    """
    Wrapper opcional para actualizaciones de red target.

    Nota:
    - En tu implementación actual, DQNAgent ya maneja self.q_tgt y soft_update().
    - Este wrapper existe por compatibilidad con código viejo y herramientas externas.
    """

    def __init__(self, target: nn.Module):
        self.net = target

    @torch.no_grad()
    def hard_update_from(self, source: nn.Module):
        self.net.load_state_dict(source.state_dict())

    @torch.no_grad()
    def soft_update_from(self, source: nn.Module, tau: float):
        tau = float(tau)
        for p_src, p_tgt in zip(source.parameters(), self.net.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p_src.data)