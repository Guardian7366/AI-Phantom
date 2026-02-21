# training/trainers/ppo_trainer.py
from __future__ import annotations

import os
import inspect
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import numpy as np
import torch

from controllers.evaluation_controller import EvaluationController
from utils.logging import ensure_dir, save_json, timestamp


@dataclass
class PPOTrainConfig:
    # Loop
    num_updates: int = 2000
    rollout_steps: int = 2048
    max_steps_per_episode: int = 128

    # Eval
    eval_every_updates: int = 20
    eval_episodes_quick: int = 120
    eval_episodes: int = 250
    eval_full_every_evals: int = 3

    # Curriculum
    success_window: int = 200
    min_samples_to_advance: int = 200
    advance_threshold: float = 0.98
    curriculum_max_level: int = 2

    # mezcla suave (0->next)
    mix_next_prob_step: float = 0.2
    mix_next_prob_cap: float = 0.8

    # interleave lower en lvl2
    interleave_lower_prob: float = 0.15
    interleave_lower_min_level: int = 1

    # IO
    run_name: str = "maze_ppo_v1"
    results_dir: str = "results/runs"
    checkpoint_dir: str = "results/checkpoints"
    seed: int = 42

    # checkpoints
    save_last_every_updates: int = 20


class PPOTrainer:
    """
    PPO trainer single-env, compatible con MazeEnvironment y EvaluationController.
    """

    def __init__(self, env, agent, *, cfg: PPOTrainConfig, device: torch.device):
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.device = device

        self.eval_ctrl = EvaluationController(env, agent)

        ensure_dir(str(self.cfg.results_dir))
        ensure_dir(str(self.cfg.checkpoint_dir))

        self._eval_accepts_freeze_pool = self._fn_accepts_kw(self.eval_ctrl.evaluate, "freeze_pool")
        self._env_reset_accepts_freeze_pool = self._fn_accepts_kw(getattr(self.env, "reset", None), "freeze_pool")
        self._env_reset_accepts_reset_pool = self._fn_accepts_kw(getattr(self.env, "reset", None), "reset_pool")

        self.curriculum_level = 0
        self.mix_next_prob = 0.0

        self.recent_by_level = {
            lvl: deque(maxlen=int(self.cfg.success_window))
            for lvl in range(int(self.cfg.curriculum_max_level) + 1)
        }

        self._eval_tick = 0
        self._last_eval_main: Optional[Dict[str, Any]] = None
        self._last_eval_lvl2: Optional[Dict[str, Any]] = None

        self.history: Dict[str, Any] = {
            "update": [],
            "steps_total": [],
            "curriculum_level_base": [],
            "mix_next_prob": [],
            "rollout": [],
            "ppo": [],
            "eval": [],
        }

    @staticmethod
    def _fn_accepts_kw(fn, kw: str) -> bool:
        try:
            sig = inspect.signature(fn)
            return kw in sig.parameters
        except Exception:
            return False

    def _recent_sr(self, lvl: int) -> float:
        q = self.recent_by_level.get(int(lvl))
        if not q or len(q) == 0:
            return 0.0
        if len(q) < int(self.cfg.min_samples_to_advance):
            return 0.0
        return float(np.mean(q))

    def _maybe_advance_curriculum(self):
        if self.curriculum_level >= int(self.cfg.curriculum_max_level):
            return
        base_sr = self._recent_sr(self.curriculum_level)
        if base_sr < float(self.cfg.advance_threshold):
            return

        if self.mix_next_prob < float(self.cfg.mix_next_prob_cap):
            self.mix_next_prob = min(
                float(self.cfg.mix_next_prob_cap),
                float(self.mix_next_prob) + float(self.cfg.mix_next_prob_step),
            )
        else:
            self.curriculum_level += 1
            self.mix_next_prob = 0.0

    def _sample_curriculum_level_for_episode(self, rng: np.random.Generator) -> int:
        max_lvl = int(self.cfg.curriculum_max_level)

        if self.curriculum_level >= max_lvl:
            p = float(self.cfg.interleave_lower_prob)
            if p > 0.0 and rng.random() < p:
                low = max(0, min(int(self.cfg.interleave_lower_min_level), max_lvl))
                return int(low)
            return int(self.curriculum_level)

        if self.mix_next_prob <= 0.0:
            return int(self.curriculum_level)
        if rng.random() < float(self.mix_next_prob):
            return int(min(self.curriculum_level + 1, max_lvl))
        return int(self.curriculum_level)

    def _env_reset_train(self, *, curriculum_level: int, seed: int):
        kwargs = {"curriculum_level": int(curriculum_level), "seed": int(seed)}
        if self._env_reset_accepts_freeze_pool:
            kwargs["freeze_pool"] = False
        if self._env_reset_accepts_reset_pool:
            kwargs["reset_pool"] = False
        return self.env.reset(**kwargs)

    def _evaluate(self, *, episodes: int, curriculum_level: int, seed: int, freeze_pool: bool = True) -> Dict[str, Any]:
        self.agent.eval_mode()
        if self._eval_accepts_freeze_pool:
            return self.eval_ctrl.evaluate(
                episodes=int(episodes),
                curriculum_level=int(curriculum_level),
                seed=int(seed),
                freeze_pool=bool(freeze_pool),
                reset_pool_on_eval=False,
            )
        return self.eval_ctrl.evaluate(
            episodes=int(episodes),
            curriculum_level=int(curriculum_level),
            seed=int(seed),
            reset_pool_on_eval=False,
        )

    def _save_checkpoint(self, folder: str, name: str):
        ensure_dir(folder)
        path = os.path.join(folder, name)
        state = self.agent.get_state_dict()
        torch.save(state, path)

    @staticmethod
    def _gae(
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
        gamma: float,
        lam: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        dones: 1.0 SOLO si terminal REAL (terminated), NO por timeout/truncated.
        """
        T = int(rewards.shape[0])
        adv = np.zeros((T,), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else float(values[t + 1])
            nonterminal = 1.0 - float(dones[t])
            delta = float(rewards[t]) + gamma * nonterminal * next_value - float(values[t])
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = float(last_gae)
        ret = adv + values.astype(np.float32, copy=False)
        return adv, ret

    def _ppo_update(
        self,
        obs_t: torch.Tensor,        # (T,3,H,W)
        actions_t: torch.Tensor,    # (T,)
        old_logp_t: torch.Tensor,   # (T,)
        adv_t: torch.Tensor,        # (T,)
        ret_t: torch.Tensor,        # (T,)
        old_val_t: torch.Tensor,    # (T,1) usado para value clipping si se activa
    ) -> Dict[str, Any]:
        cfg = self.agent.cfg

        clip_eps = float(cfg.clip_eps)
        ent_coef = float(cfg.entropy_coef)
        v_coef = float(cfg.value_coef)
        max_gn = float(cfg.max_grad_norm)

        # ✅ KL early stop
        target_kl = float(getattr(cfg, "target_kl", 0.0) or 0.0)
        target_kl_mult = float(getattr(cfg, "target_kl_multiplier", 1.5) or 1.5)
        kl_stop_threshold = target_kl * target_kl_mult if target_kl > 0.0 else None

        # ✅ value clipping
        clip_value_loss = bool(getattr(cfg, "clip_value_loss", True))
        value_clip_eps = float(getattr(cfg, "value_clip_eps", clip_eps))

        # normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std().clamp(min=1e-8))

        T = int(obs_t.shape[0])
        mb = int(cfg.minibatch_size)
        mb = max(1, min(mb, T))

        autocast_enabled = bool(self.agent.scaler.is_enabled())

        losses_pi: List[float] = []
        losses_v: List[float] = []
        entropies: List[float] = []
        kl_approx: List[float] = []
        clip_frac: List[float] = []

        idxs = torch.randperm(T, device=self.device)

        early_stop = False
        epochs_ran = 0

        for _epoch in range(int(cfg.ppo_epochs)):
            epochs_ran += 1
            for start in range(0, T, mb):
                j = idxs[start:start + mb]

                b_obs = obs_t[j]
                b_act = actions_t[j]
                b_old_logp = old_logp_t[j]
                b_adv = adv_t[j]
                b_ret = ret_t[j]
                b_old_val = old_val_t[j].squeeze(1)

                with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    logp, ent, v = self.agent.evaluate_actions(b_obs, b_act)
                    ratio = torch.exp(logp - b_old_logp)

                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                    loss_pi = -torch.min(surr1, surr2).mean()

                    v_pred = v.squeeze(1)

                    if clip_value_loss:
                        v_clipped = b_old_val + torch.clamp(v_pred - b_old_val, -value_clip_eps, value_clip_eps)
                        v_loss_1 = (b_ret - v_pred).pow(2)
                        v_loss_2 = (b_ret - v_clipped).pow(2)
                        loss_v = 0.5 * torch.max(v_loss_1, v_loss_2).mean()
                    else:
                        loss_v = 0.5 * (b_ret - v_pred).pow(2).mean()

                    loss_ent = ent.mean()
                    loss = loss_pi + v_coef * loss_v - ent_coef * loss_ent

                if not torch.isfinite(loss):
                    self.agent.optim.zero_grad(set_to_none=True)
                    continue

                self.agent.optim.zero_grad(set_to_none=True)
                self.agent.scaler.scale(loss).backward()
                self.agent.scaler.unscale_(self.agent.optim)
                torch.nn.utils.clip_grad_norm_(self.agent.net.parameters(), max_gn)
                self.agent.scaler.step(self.agent.optim)
                self.agent.scaler.update()

                with torch.no_grad():
                    approx_kl = (b_old_logp - logp).mean()
                    cf = (torch.abs(ratio - 1.0) > clip_eps).float().mean()

                losses_pi.append(float(loss_pi.item()))
                losses_v.append(float(loss_v.item()))
                entropies.append(float(loss_ent.item()))
                kl_approx.append(float(approx_kl.item()))
                clip_frac.append(float(cf.item()))

                # ✅ EARLY STOP por KL (evita “policy collapse”)
                if kl_stop_threshold is not None and float(approx_kl.item()) > float(kl_stop_threshold):
                    early_stop = True
                    break

            if early_stop:
                break

        return {
            "loss_pi": float(np.mean(losses_pi)) if losses_pi else None,
            "loss_v": float(np.mean(losses_v)) if losses_v else None,
            "entropy": float(np.mean(entropies)) if entropies else None,
            "approx_kl": float(np.mean(kl_approx)) if kl_approx else None,
            "clip_frac": float(np.mean(clip_frac)) if clip_frac else None,
            "kl_early_stop": bool(early_stop),
            "epochs_ran": int(epochs_ran),
            "kl_stop_threshold": float(kl_stop_threshold) if kl_stop_threshold is not None else None,
        }

    def train(self) -> Dict[str, Any]:
        rng = np.random.default_rng(int(self.cfg.seed))

        # Lock pool eval lvl2 una vez (si aplica)
        if int(self.cfg.curriculum_max_level) >= 2 and self._env_reset_accepts_reset_pool:
            kwargs = {"curriculum_level": 2, "seed": int(self.cfg.seed), "reset_pool": True}
            if self._env_reset_accepts_freeze_pool:
                kwargs["freeze_pool"] = True
            try:
                _ = self.env.reset(**kwargs)
            except Exception:
                pass

        run_id = f"{self.cfg.run_name}_{timestamp()}_seed{int(self.cfg.seed)}"
        run_path = os.path.join(str(self.cfg.checkpoint_dir), run_id)
        ensure_dir(run_path)

        best_eval_sr_main = -1.0
        best_eval_sr_lvl2 = -1.0

        steps_total = 0

        # Estado episodio actual
        self.agent.train_mode()
        lvl_ep = 0
        ep_seed = int(rng.integers(0, 10_000_000))
        obs, _info0 = self._env_reset_train(curriculum_level=lvl_ep, seed=ep_seed)
        ep_steps = 0
        ep_success = 0
        ep_return = 0.0

        for upd in range(1, int(self.cfg.num_updates) + 1):
            self.agent.train_mode()

            T = int(self.cfg.rollout_steps)

            obs_buf: List[np.ndarray] = []
            act_buf: List[int] = []
            logp_buf: List[float] = []
            val_buf: List[float] = []
            rew_buf: List[float] = []

            # ✅ IMPORTANT: dones para GAE = terminal REAL (terminated), NO truncated
            done_for_gae_buf: List[float] = []
            truncated_buf: List[float] = []  # solo logging

            # rollout metrics
            roll_eps = 0
            roll_ok = 0
            roll_return_sum = 0.0
            roll_steps = 0
            roll_invalid_sum = 0
            roll_revisit_sum = 0
            roll_unique_sum = 0

            ep_lens: List[int] = []
            ep_rets: List[float] = []

            for _t in range(T):
                # Si el episodio ya acabó, reseteamos aquí
                if ep_steps == 0:
                    lvl_ep = self._sample_curriculum_level_for_episode(rng)
                    ep_seed = int(rng.integers(0, 10_000_000))
                    obs, _info0 = self._env_reset_train(curriculum_level=int(lvl_ep), seed=int(ep_seed))
                    ep_success = 0
                    ep_return = 0.0

                a, logp, v = self.agent.act_with_logprob_value(obs)
                next_obs, r, terminated, truncated, info = self.env.step(int(a))

                terminated = bool(terminated)
                truncated = bool(truncated)

                # ✅ terminal real para GAE
                done_for_gae = 1.0 if terminated else 0.0
                terminal_any = bool(terminated or truncated)

                obs_buf.append(np.asarray(obs, dtype=np.float32))
                act_buf.append(int(a))
                logp_buf.append(float(logp))
                val_buf.append(float(v))
                rew_buf.append(float(r))
                done_for_gae_buf.append(float(done_for_gae))
                truncated_buf.append(1.0 if truncated else 0.0)

                # metrics
                roll_return_sum += float(r)
                roll_steps += 1
                ep_steps += 1
                ep_return += float(r)

                if isinstance(info, dict):
                    roll_invalid_sum += int(info.get("invalid_moves", 0))
                    roll_revisit_sum += int(info.get("revisit_steps", 0))
                    roll_unique_sum += int(info.get("visited_unique", 0))
                    if bool(info.get("reached_goal", False)):
                        ep_success = 1

                obs = next_obs

                if terminal_any or ep_steps >= int(self.cfg.max_steps_per_episode):
                    roll_eps += 1
                    roll_ok += int(ep_success)

                    ep_lens.append(int(ep_steps))
                    ep_rets.append(float(ep_return))

                    self.recent_by_level[int(lvl_ep)].append(int(ep_success))
                    self._maybe_advance_curriculum()

                    # fuerza nuevo episodio
                    ep_steps = 0

            # bootstrap value del último obs (para el último paso del rollout)
            with torch.no_grad():
                x_last = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                v_last = float(self.agent.get_value(x_last).squeeze(1).item())

            rewards = np.asarray(rew_buf, dtype=np.float32)
            dones_for_gae = np.asarray(done_for_gae_buf, dtype=np.float32)
            values = np.asarray(val_buf, dtype=np.float32)

            adv, ret = self._gae(
                rewards=rewards,
                dones=dones_for_gae,
                values=values,
                last_value=v_last,
                gamma=float(self.agent.cfg.gamma),
                lam=float(self.agent.cfg.gae_lambda),
            )

            obs_t = torch.as_tensor(np.stack(obs_buf, axis=0), dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(np.asarray(act_buf, dtype=np.int64), dtype=torch.long, device=self.device)
            old_logp_t = torch.as_tensor(np.asarray(logp_buf, dtype=np.float32), dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
            ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
            old_val_t = torch.as_tensor(values.reshape(-1, 1), dtype=torch.float32, device=self.device)

            ppo_stats = self._ppo_update(obs_t, actions_t, old_logp_t, adv_t, ret_t, old_val_t)

            steps_total += int(T)
            self.agent.total_steps = int(self.agent.total_steps + T)

            roll_sr = float(roll_ok / max(1, roll_eps))
            roll_r_mean = float(roll_return_sum / max(1, roll_steps))
            base_sr = float(self._recent_sr(self.curriculum_level))

            self.history["update"].append(int(upd))
            self.history["steps_total"].append(int(steps_total))
            self.history["curriculum_level_base"].append(int(self.curriculum_level))
            self.history["mix_next_prob"].append(float(self.mix_next_prob))
            self.history["rollout"].append({
                "episodes": int(roll_eps),
                "success_rate": float(roll_sr),
                "reward_mean_per_step": float(roll_r_mean),
                "mean_ep_len": float(np.mean(ep_lens)) if ep_lens else None,
                "mean_ep_return": float(np.mean(ep_rets)) if ep_rets else None,
                "invalid_moves_sum": int(roll_invalid_sum),
                "revisit_steps_sum": int(roll_revisit_sum),
                "visited_unique_sum": int(roll_unique_sum),
                "recent_sr_base_gated": float(base_sr),
                "truncated_frac": float(np.mean(np.asarray(truncated_buf, dtype=np.float32))) if truncated_buf else 0.0,
            })
            self.history["ppo"].append(dict(ppo_stats))

            if upd % 5 == 0:
                print(
                    f"[UPD {upd:5d}] lvl_base={self.curriculum_level} mix={self.mix_next_prob:.2f} "
                    f"roll_SR={roll_sr:.3f} recent_SR_gated={base_sr:.3f} "
                    f"pi={ppo_stats.get('loss_pi')} v={ppo_stats.get('loss_v')} "
                    f"ent={ppo_stats.get('entropy')} kl={ppo_stats.get('approx_kl')} "
                    f"clip={ppo_stats.get('clip_frac')} kl_stop={ppo_stats.get('kl_early_stop')}"
                )

            # eval
            if upd % int(self.cfg.eval_every_updates) == 0:
                self._eval_tick += 1

                eval_level_main = int(
                    self.cfg.curriculum_max_level
                    if self.curriculum_level >= int(self.cfg.curriculum_max_level)
                    else self.curriculum_level
                )
                eval_seed = int(rng.integers(0, 10_000_000))
                eval_stats_main = self._evaluate(
                    episodes=int(self.cfg.eval_episodes_quick),
                    curriculum_level=int(eval_level_main),
                    seed=int(eval_seed),
                    freeze_pool=True,
                )
                self._last_eval_main = dict(eval_stats_main)

                self.history["eval"].append({
                    "update": int(upd),
                    "eval_level": int(eval_level_main),
                    "eval_seed": int(eval_seed),
                    "tag": "quick",
                    **eval_stats_main,
                })

                sr_main = float(eval_stats_main.get("success_rate", 0.0))
                if sr_main > best_eval_sr_main:
                    best_eval_sr_main = sr_main
                    self._save_checkpoint(run_path, "best_model.pth")

                eval_stats_lvl2 = None
                do_full = (self._eval_tick % int(self.cfg.eval_full_every_evals) == 0)
                if do_full and int(self.cfg.curriculum_max_level) >= 2:
                    eval_seed2 = int(rng.integers(0, 10_000_000))
                    eval_stats_lvl2 = self._evaluate(
                        episodes=int(self.cfg.eval_episodes),
                        curriculum_level=int(self.cfg.curriculum_max_level),
                        seed=int(eval_seed2),
                        freeze_pool=True,
                    )
                    self._last_eval_lvl2 = dict(eval_stats_lvl2)

                    self.history["eval"].append({
                        "update": int(upd),
                        "eval_level": int(self.cfg.curriculum_max_level),
                        "eval_seed": int(eval_seed2),
                        "tag": "full_lvl2",
                        **eval_stats_lvl2,
                    })

                    sr_lvl2 = float(eval_stats_lvl2.get("success_rate", 0.0))
                    if sr_lvl2 > best_eval_sr_lvl2:
                        best_eval_sr_lvl2 = sr_lvl2
                        self._save_checkpoint(run_path, "best_model_lvl2.pth")

                if upd % int(self.cfg.save_last_every_updates) == 0:
                    self._save_checkpoint(run_path, "last_model.pth")

                payload: Dict[str, Any] = {
                    "run_id": str(run_id),
                    "update": int(upd),
                    "steps_total": int(steps_total),
                    "curriculum_level": int(self.curriculum_level),
                    "mix_next_prob": float(self.mix_next_prob),
                    "best_eval_success_rate": float(best_eval_sr_main),
                    "best_eval_success_rate_lvl2": float(best_eval_sr_lvl2),
                    "last_eval_quick": {"eval_level": int(eval_level_main), **eval_stats_main},
                }
                if eval_stats_lvl2 is not None:
                    payload["last_eval_full_lvl2"] = {
                        "eval_level": int(self.cfg.curriculum_max_level),
                        **eval_stats_lvl2,
                    }

                save_json(os.path.join(str(self.cfg.results_dir), f"{run_id}.json"), payload)

                print(
                    f"[EVAL UPD {upd:5d}] lvl={eval_level_main} SR={float(eval_stats_main.get('success_rate', 0.0)):.3f} "
                    f"ratio={float(eval_stats_main.get('mean_ratio_vs_bfs_start', float('nan'))):.2f}"
                )

        save_json(os.path.join(str(self.cfg.results_dir), f"{run_id}_history.json"), self.history)

        return {
            "run_id": str(run_id),
            "checkpoint_path": str(run_path),
            "best_eval_success_rate": float(best_eval_sr_main),
            "best_eval_success_rate_lvl2": float(best_eval_sr_lvl2),
        }