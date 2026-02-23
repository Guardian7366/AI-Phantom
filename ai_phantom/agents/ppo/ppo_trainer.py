# ai_phantom/agents/ppo/ppo_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .model import CnnActorCritic
from .buffer import RolloutBuffer
from .action_mask import mask_invalid_actions


@dataclass
class PPOConfig:
    rollout_len: int = 128
    lr: float = 1.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    max_grad_norm: float = 0.5

    ppo_epochs: int = 4
    minibatch_size: int = 64

    # KL target
    target_kl: float = 0.03
    vf_clip_range: float = 0.2

    enable_action_mask: bool = True

    # Protecciones numéricas
    abort_on_nan: bool = True
    nan_logits_replacement: float = 0.0

    # ------------------------------
    # LR/clip adaptativo por KL (conservador)
    # ------------------------------
    adaptive_kl: bool = True

    kl_low_mult: float = 0.5
    kl_high_mult: float = 1.5
    kl_ema_beta: float = 0.90

    lr_min: float = 2.5e-5
    lr_max: float = 2.0e-4

    clip_min: float = 0.10
    clip_max: float = 0.25

    lr_down_factor: float = 0.75
    lr_up_factor: float = 1.05

    clip_down_factor: float = 0.90
    clip_up_factor: float = 1.02

    # ------------------------------
    # ✅ Fix extra: estabilidad de ratio / logp
    # ------------------------------
    ratio_clip_max: float = 5.0           # evita explosiones si logp se va
    ratio_nan_replacement: float = 1.0    # neutral

    # Aux imitation (solo teacher steps)
    bc_coef: float = 0.02
    bc_coef_end: float = 0.0
    bc_decay_updates: int = 200


class PPOTrainer:
    """
    PPO Trainer (single-env) con:
    - action masking consistente (env obs -> invalid actions a -inf)
    - saneamiento de logits (preserva -inf, corrige NaN/+inf)
    - clipping de ratio seguro
    - explained variance estable
    - LR/clip adaptativo por EMA(|KL|)
    - early-stop por KL
    """
    def __init__(self, model: CnnActorCritic, cfg: PPOConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), lr=float(cfg.lr), eps=1e-5)
        self.updates = 0

        self._kl_ema: Optional[float] = None

    @staticmethod
    def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        # y_true puede ser casi constante en entornos sencillos; maneja var ~ 0
        var_y = torch.var(y_true)
        if var_y.item() < 1e-12:
            return 0.0
        ev = 1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        return float(ev.clamp(-1.0, 1.0).item())

    def _sanitize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mantiene -inf (del masking). Solo corrige NaN/+inf.
        Si una fila queda completamente inválida (todo -inf), la rescata con ceros.
        """
        logits = torch.nan_to_num(
            logits,
            nan=float(self.cfg.nan_logits_replacement),
            posinf=float(self.cfg.nan_logits_replacement),
            neginf=float("-inf"),
        )

        all_neginf = torch.isneginf(logits).all(dim=-1)
        if bool(all_neginf.any()):
            logits = logits.clone()
            logits[all_neginf] = 0.0
        return logits

    def _get_lr(self) -> float:
        return float(self.optim.param_groups[0]["lr"])

    def _set_lr_clamped(self, lr: float) -> None:
        lr = float(lr)
        lr = max(float(self.cfg.lr_min), min(float(self.cfg.lr_max), lr))
        self.optim.param_groups[0]["lr"] = lr

    def _set_clip_clamped(self, clip: float) -> None:
        clip = float(clip)
        clip = max(float(self.cfg.clip_min), min(float(self.cfg.clip_max), clip))
        self.cfg.clip_range = clip

    def _update_kl_ema(self, kl_mag: float) -> float:
        kl_mag = float(kl_mag)
        beta = float(self.cfg.kl_ema_beta)
        if self._kl_ema is None:
            self._kl_ema = kl_mag
        else:
            self._kl_ema = beta * self._kl_ema + (1.0 - beta) * kl_mag
        return float(self._kl_ema)

    def _adaptive_step(self, kl_ema: float) -> None:
        """
        Ajuste conservador basado en EMA(|KL|).
        - si KL alto -> baja lr y clip
        - si KL bajo  -> sube lr y clip lentamente
        """
        if not bool(self.cfg.adaptive_kl):
            return

        target = float(self.cfg.target_kl)
        if target <= 0.0:
            return

        low = target * float(self.cfg.kl_low_mult)
        high = target * float(self.cfg.kl_high_mult)

        lr = self._get_lr()
        clip = float(self.cfg.clip_range)

        if kl_ema > high:
            lr = lr * float(self.cfg.lr_down_factor)
            clip = clip * float(self.cfg.clip_down_factor)
            self._set_lr_clamped(lr)
            self._set_clip_clamped(clip)
            return

        if kl_ema < low:
            lr = lr * float(self.cfg.lr_up_factor)
            clip = clip * float(self.cfg.clip_up_factor)
            self._set_lr_clamped(lr)
            self._set_clip_clamped(clip)
            return

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        self.updates += 1
        vf_coef = float(self.cfg.vf_coef)
        ent_coef = float(self.cfg.ent_coef)
        max_gn = float(self.cfg.max_grad_norm)
        target_kl = float(self.cfg.target_kl)
        vf_clip = float(self.cfg.vf_clip_range)

        pi_loss_acc = 0.0
        vf_loss_acc = 0.0
        ent_acc = 0.0
        bc_loss_acc = 0.0
        kl_mag_acc = 0.0
        kl_signed_acc = 0.0
        ev_acc = 0.0
        n_batches = 0

        self.model.train()

        early_stop = False
        nan_abort = False

        last_kl_ema = float(self._kl_ema) if (self._kl_ema is not None) else 0.0

        # Nota: no uses autocast aquí a menos que estés 100% seguro; PPO con logits puede ser sensible.
        for _epoch in range(int(self.cfg.ppo_epochs)):
            for batch in buffer.iter_minibatches(self.cfg.minibatch_size, shuffle=True):
                logits, value = self.model(batch.obs)

                # Action masking + saneamiento
                logits = mask_invalid_actions(batch.obs, logits, enable=bool(self.cfg.enable_action_mask))
                logits = self._sanitize_logits(logits)

                # value: asegúrate de shape [B]
                if value.dim() == 2 and value.size(-1) == 1:
                    value = value.squeeze(-1)

                # Abort por NaNs en value (antes de pérdidas)
                if bool(self.cfg.abort_on_nan) and (not torch.isfinite(value).all()):
                    nan_abort = True
                    break

                dist = Categorical(logits=logits)
                logp = dist.log_prob(batch.actions)
                entropy = dist.entropy().mean()

                # Si algo raro en logp -> abort
                if bool(self.cfg.abort_on_nan) and (not torch.isfinite(logp).all()):
                    nan_abort = True
                    break

                clip = float(self.cfg.clip_range)

                # ✅ ratio estable
                logp_diff = (logp - batch.logp_old)
                if bool(self.cfg.abort_on_nan) and (not torch.isfinite(logp_diff).all()):
                    nan_abort = True
                    break

                ratio = torch.exp(logp_diff)
                ratio = torch.nan_to_num(
                    ratio,
                    nan=float(self.cfg.ratio_nan_replacement),
                    posinf=float(self.cfg.ratio_nan_replacement),
                    neginf=float(self.cfg.ratio_nan_replacement),
                ).clamp(0.0, float(self.cfg.ratio_clip_max))

                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * batch.advantages
                pi_loss = -torch.min(surr1, surr2).mean()

                # ✅ Value function clipping (PPO2 style)
                v_pred = value
                v_old = batch.values_old
                v_clipped = v_old + torch.clamp(v_pred - v_old, -vf_clip, vf_clip)

                vf_loss1 = F.mse_loss(v_pred, batch.returns)
                vf_loss2 = F.mse_loss(v_clipped, batch.returns)

                vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)

                loss = pi_loss + vf_coef * vf_loss - ent_coef * entropy

                # -------------------------------------------------
                # ✅ BC loss solo en pasos teacher (si existen)
                # -------------------------------------------------
                bc_loss = torch.tensor(0.0, device=self.device)

                if hasattr(batch, "is_teacher") and batch.is_teacher is not None:
                    mask = (batch.is_teacher > 0.5)

                    if bool(mask.any()):
                        t = min(
                            1.0,
                            float(self.updates - 1)
                            / float(max(1, self.cfg.bc_decay_updates)),
                        )

                        bc_coef = (
                            (1.0 - t) * float(self.cfg.bc_coef)
                            + t * float(self.cfg.bc_coef_end)
                        )

                        bc_loss = F.cross_entropy(
                            logits[mask],
                            batch.actions[mask],
                        )

                        loss = loss + float(bc_coef) * bc_loss

                if bool(self.cfg.abort_on_nan) and (not torch.isfinite(loss).all()):
                    nan_abort = True
                    break

                self.optim.zero_grad(set_to_none=True)
                loss.backward()

                # ✅ evita NaNs por grad explosivo
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_gn)
                self.optim.step()

                with torch.no_grad():
                    # ✅ KL: usa magnitud |KL| para control, y conserva signed para debug
                    approx_kl_signed = (batch.logp_old - logp).mean()
                    approx_kl_mag = approx_kl_signed.abs()
                    ev = self._explained_variance(v_pred, batch.returns)

                pi_loss_acc += float(pi_loss.item())
                vf_loss_acc += float(vf_loss.item())
                ent_acc += float(entropy.item())
                bc_loss_acc += float(bc_loss.item())
                kl_mag_acc += float(approx_kl_mag.item())
                kl_signed_acc += float(approx_kl_signed.item())
                ev_acc += float(ev)
                n_batches += 1

                last_kl_ema = self._update_kl_ema(float(approx_kl_mag.item()))
                self._adaptive_step(last_kl_ema)

                # ✅ Early-stop por |KL| (EMA)
                if target_kl > 0.0 and float(last_kl_ema) > (target_kl * 1.5):
                    early_stop = True
                    break

            if nan_abort or early_stop:
                break

        if n_batches == 0:
            return {
                "pi_loss": 0.0,
                "vf_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,             # |KL| promedio batches
                "approx_kl_signed": 0.0,      # KL con signo (debug)
                "explained_var": 0.0,
                "early_stop": 1.0 if early_stop else 0.0,
                "nan_abort": 1.0 if nan_abort else 0.0,
                "lr": float(self._get_lr()),
                "clip": float(self.cfg.clip_range),
                "kl_ema": float(last_kl_ema),
            }

        return {
            "pi_loss": pi_loss_acc / n_batches,
            "vf_loss": vf_loss_acc / n_batches,
            "entropy": ent_acc / n_batches,
            "approx_kl": kl_mag_acc / n_batches,             # |KL|
            "approx_kl_signed": kl_signed_acc / n_batches,   # debug
            "explained_var": ev_acc / n_batches,
            "early_stop": 1.0 if early_stop else 0.0,
            "nan_abort": 1.0 if nan_abort else 0.0,
            "lr": float(self._get_lr()),
            "clip": float(self.cfg.clip_range),
            "kl_ema": float(last_kl_ema),
            "bc_loss": bc_loss_acc / n_batches,
        }