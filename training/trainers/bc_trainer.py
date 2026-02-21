# training/trainers/bc_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class BCConfig:
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 5
    grad_clip_norm: float = 5.0
    use_amp: bool = True
    num_workers: int = 0
    shuffle: bool = True
    seed: int = 0


class ExpertDataset(Dataset):
    def __init__(self, obs: np.ndarray, actions: np.ndarray):
        assert obs.ndim == 4, "obs debe ser (N,3,H,W)"
        assert actions.ndim == 1, "actions debe ser (N,)"
        assert obs.shape[0] == actions.shape[0], "N inconsistente"
        self.obs = obs.astype(np.float32, copy=False)
        self.actions = actions.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, idx: int):
        return self.obs[idx], self.actions[idx]


class BCTrainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[BCConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg or BCConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        torch.manual_seed(int(self.cfg.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.cfg.seed))

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        use_amp = bool(self.cfg.use_amp and self.device.type == "cuda")
        if use_amp:
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)

    @staticmethod
    def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path)
        obs = data["obs"]
        actions = data["actions"]
        return obs, actions

    def fit_from_npz(self, npz_path: str, *, save_path: Optional[str] = None) -> Dict[str, Any]:
        obs, actions = self.load_npz(npz_path)
        return self.fit(obs, actions, save_path=save_path)

    def fit(self, obs: np.ndarray, actions: np.ndarray, *, save_path: Optional[str] = None) -> Dict[str, Any]:
        ds = ExpertDataset(obs, actions)
        dl = DataLoader(
            ds,
            batch_size=int(self.cfg.batch_size),
            shuffle=bool(self.cfg.shuffle),
            num_workers=int(self.cfg.num_workers),
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        self.model.train()
        autocast_enabled = bool(self.scaler.is_enabled())

        total = 0
        correct = 0
        loss_sum = 0.0

        for epoch in range(int(self.cfg.epochs)):
            ep_total = 0
            ep_correct = 0
            ep_loss_sum = 0.0

            for x_np, y_np in dl:
                x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)
                y = torch.as_tensor(y_np, dtype=torch.long, device=self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    logits = self.model(x)  # (B,num_actions) — Q-values como logits
                    loss = self.loss_fn(logits, y)

                if not torch.isfinite(loss):
                    self.optim.zero_grad(set_to_none=True)
                    continue

                self.optim.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
                self.scaler.step(self.optim)
                self.scaler.update()

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    ep_total += int(y.shape[0])
                    ep_correct += int((pred == y).sum().item())
                    ep_loss_sum += float(loss.item()) * int(y.shape[0])

            # acumula stats globales
            total += ep_total
            correct += ep_correct
            loss_sum += ep_loss_sum

        out = {
            "samples": int(total),
            "acc": float(correct / max(1, total)),
            "loss_mean": float(loss_sum / max(1, total)),
        }

        if save_path is not None:
            torch.save(self.model.state_dict(), save_path)
            out["saved_to"] = str(save_path)

        return out