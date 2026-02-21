# agents/ppo/ppo_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from agents.ppo.networks import ActorCriticNetwork


@dataclass
class PPOConfig:
    num_actions: int = 4

    # RL core
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO objective
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # ✅ Stability knobs (nuevo)
    target_kl: float = 0.02              # early stop si KL se pasa
    target_kl_multiplier: float = 1.5    # umbral real = target_kl * multiplier

    # ✅ Value clipping (nuevo)
    clip_value_loss: bool = True
    value_clip_eps: float = 0.2

    # Optim
    lr: float = 2.5e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5
    use_amp: bool = True

    # Update
    ppo_epochs: int = 4
    minibatch_size: int = 256

    # Repro
    seed: int = 0


class PPOAgent:
    """
    PPO Agent compatible con EvaluationController:
    - act(obs, deterministic=False) -> int
    Además expone:
    - act_with_logprob_value(obs) -> (a, logp, v)
    - evaluate_actions(obs_batch, action_batch) -> logp, entropy, value
    - get_value(obs_batch) -> value
    - train_mode / eval_mode
    - get_state_dict / load_state_dict (snapshot-friendly)
    """
    def __init__(self, cfg: Optional[PPOConfig] = None, device: Optional[torch.device] = None):
        self.cfg = cfg or PPOConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = int(getattr(self.cfg, "seed", 0))
        self.rng = np.random.default_rng(seed)

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.net = ActorCriticNetwork(in_channels=3, num_actions=int(self.cfg.num_actions)).to(self.device)

        self.optim = optim.AdamW(
            self.net.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        if use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)

        self.total_steps = 0
        self.last_stats: Dict[str, Any] = {}

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,3,H,W)
        logits, _v = self.net(x)
        if deterministic:
            return int(torch.argmax(logits, dim=1).item())

        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item())

    @torch.no_grad()
    def act_with_logprob_value(self, obs: np.ndarray) -> Tuple[int, float, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, v = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)  # (1,)
        return int(a.item()), float(logp.item()), float(v.squeeze(1).item())

    def _forward_dist_value(self, obs_t: torch.Tensor):
        logits, v = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, v

    def evaluate_actions(self, obs_t: torch.Tensor, actions_t: torch.Tensor):
        """
        obs_t: (B,3,H,W) float32
        actions_t: (B,) int64
        -> logp(B,), entropy(B,), value(B,1)
        """
        dist, v = self._forward_dist_value(obs_t)
        logp = dist.log_prob(actions_t)
        ent = dist.entropy()
        return logp, ent, v

    @torch.no_grad()
    def get_value(self, obs_t: torch.Tensor) -> torch.Tensor:
        _dist, v = self._forward_dist_value(obs_t)
        return v

    # ---- Snapshot / reproducibilidad ----
    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "net": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "optim": self.optim.state_dict(),
            "total_steps": int(self.total_steps),
            "rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        sd = {k: v.to(self.device) for k, v in state["net"].items()}
        self.net.load_state_dict(sd)

        self.optim.load_state_dict(state["optim"])
        for st in self.optim.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(self.device)

        self.total_steps = int(state.get("total_steps", 0))

        self.rng = np.random.default_rng(0)
        self.rng.bit_generator.state = state["rng_state"]

        if state.get("torch_rng_state", None) is not None:
            torch.set_rng_state(state["torch_rng_state"])
        if self.device.type == "cuda" and state.get("cuda_rng_state_all", None) is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])