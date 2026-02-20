from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from agents.dqn.dqn_agent import DQNAgent, DQNConfig


@dataclass
class ModelLoadResult:
    model_path: str
    device: torch.device


def resolve_device(device_str: str) -> torch.device:
    s = (device_str or "auto").lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s in ("cuda", "cpu"):
        return torch.device(s)
    raise ValueError("device debe ser: 'auto', 'cuda' o 'cpu'.")


def resolve_model_path(checkpoint_dir: str, which: str = "best") -> str:
    which = (which or "best").lower()
    if which not in ("best", "last"):
        raise ValueError("which debe ser 'best' o 'last'.")

    fname = "best_model.pth" if which == "best" else "last_model.pth"
    path = os.path.join(checkpoint_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el modelo: {path}")
    return path


def load_dqn_agent(
    *,
    checkpoint_dir: str,
    which: str = "best",
    device_str: str = "auto",
    agent_cfg: Optional[DQNConfig] = None,
) -> Tuple[DQNAgent, ModelLoadResult]:
    """
    Carga un DQNAgent y le mete el state_dict del modelo.
    Mantiene sincronía con DQNConfig (por default usa el DQNConfig() actual).
    """
    device = resolve_device(device_str)
    cfg = agent_cfg or DQNConfig()
    agent = DQNAgent(cfg, device=device)

    model_path = resolve_model_path(checkpoint_dir, which)
    sd = torch.load(model_path, map_location=device)

    agent.q.load_state_dict(sd)
    agent.q_tgt.load_state_dict(sd)
    agent.q.eval()
    agent.q_tgt.eval()

    return agent, ModelLoadResult(model_path=model_path, device=device)
