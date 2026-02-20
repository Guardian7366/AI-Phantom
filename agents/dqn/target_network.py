import torch
import torch.nn as nn


class TargetNetwork:
    def __init__(self, online: nn.Module):
        self.net = online

    @torch.no_grad()
    def hard_update_from(self, source: nn.Module):
        self.net.load_state_dict(source.state_dict())

    @torch.no_grad()
    def soft_update_from(self, source: nn.Module, tau: float):
        for p, pt in zip(source.parameters(), self.net.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)
