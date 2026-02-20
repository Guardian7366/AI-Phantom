# controllers/training_controller.py
from __future__ import annotations

import os
import inspect
from typing import Dict, Any, Optional
from collections import deque

import numpy as np
import torch

from controllers.evaluation_controller import EvaluationController
from utils.logging import ensure_dir, save_json, timestamp


class TrainingController:
    """
    TrainingController con:
    - Curriculum con mezcla suave (mix_next_prob)
    - Rescate en lvl2 (epsilon boost temporal)
    - Best checkpoint REAL para lvl2: best_model_lvl2.pth
    - Cuando curriculum_level == curriculum_max_level, evalúa SIEMPRE ese nivel (lvl2)
      para evitar best engañoso de lvl0/lvl1.
    - Eval defensiva: usa freeze_pool si existe en EvaluationController.evaluate
    - Modo train/eval del agente si expone train_mode/eval_mode, y fallback directo a q.train()/q.eval()
    """

    def __init__(
        self,
        env,
        agent,
        *,
        num_episodes: int,
        max_steps_per_episode: int,
        run_name: str = "maze_dqn_v3",
        results_dir: str = "results/runs",
        checkpoint_dir: str = "results/checkpoints",
        seed: int = 42,
        eval_every_episodes: int = 50,
        eval_episodes: int = 200,
        success_window: int = 200,
        advance_threshold: float = 0.98,
        curriculum_max_level: int = 2,
        # ---- anti “pozo” al avanzar ----
        epsilon_reset_on_advance: bool = True,
        epsilon_reset_value: float = 0.30,
        epsilon_reset_steps: int = 25_000,
        reset_replay_on_advance: bool = False,
        # ---- rescate lvl2 si colapsa ----
        rescue_on_stuck: bool = True,
        rescue_level: int = 2,
        rescue_sr_threshold: float = 0.55,
        rescue_patience_episodes: int = 200,
        rescue_epsilon_value: float = 0.30,
        rescue_epsilon_steps: int = 25_000,
        rescue_cooldown_episodes: int = 300,
    ):
        self.env = env
        self.agent = agent

        self.num_episodes = int(num_episodes)
        self.max_steps = int(max_steps_per_episode)

        self.run_name = str(run_name)
        self.results_dir = str(results_dir)
        self.checkpoint_dir = str(checkpoint_dir)
        self.seed = int(seed)

        self.eval_every_episodes = int(eval_every_episodes)
        self.eval_episodes = int(eval_episodes)

        self.success_window = int(success_window)
        self.advance_threshold = float(advance_threshold)
        self.curriculum_max_level = int(curriculum_max_level)

        # Anti-pozo (advance)
        self.epsilon_reset_on_advance = bool(epsilon_reset_on_advance)
        self.epsilon_reset_value = float(epsilon_reset_value)
        self.epsilon_reset_steps = int(epsilon_reset_steps)
        self.reset_replay_on_advance = bool(reset_replay_on_advance)

        # Rescate
        self.rescue_on_stuck = bool(rescue_on_stuck)
        self.rescue_level = int(rescue_level)
        self.rescue_sr_threshold = float(rescue_sr_threshold)
        self.rescue_patience_episodes = int(rescue_patience_episodes)
        self.rescue_epsilon_value = float(rescue_epsilon_value)
        self.rescue_epsilon_steps = int(rescue_epsilon_steps)
        self.rescue_cooldown_episodes = int(rescue_cooldown_episodes)

        self._rescue_bad_count = 0
        self._rescue_cooldown_left = 0

        self.eval_ctrl = EvaluationController(env, agent)

        ensure_dir(self.results_dir)
        ensure_dir(self.checkpoint_dir)

        self.history: Dict[str, Any] = {
            "episode": [],
            "loss": [],
            "epsilon": [],
            "reward": [],
            "success": [],
            "curriculum_level_base": [],
            "curriculum_level_episode": [],
            "mix_next_prob": [],
            "recent_sr_base": [],
            "eval": [],
            "rescues": [],
        }

        # Nivel “base” del curriculum
        self.curriculum_level = 0
        # Probabilidad de muestrear el siguiente nivel para mezcla suave
        self.mix_next_prob = 0.0

        # Ventanas de éxito por nivel
        self.recent_by_level = {
            lvl: deque(maxlen=self.success_window)
            for lvl in range(self.curriculum_max_level + 1)
        }

        # Cache: si evaluate() acepta freeze_pool
        self._eval_accepts_freeze_pool = self._fn_accepts_kw(self.eval_ctrl.evaluate, "freeze_pool")

    @staticmethod
    def _fn_accepts_kw(fn, kw: str) -> bool:
        try:
            sig = inspect.signature(fn)
            return kw in sig.parameters
        except Exception:
            return False

    # -------------------------
    # Agent mode helpers (no espagueti)
    # -------------------------
    def _agent_train_mode(self):
        # preferimos API explícita si existe
        if hasattr(self.agent, "train_mode"):
            self.agent.train_mode()
            return
        # fallback directo a torch modules si existen
        if hasattr(self.agent, "q") and hasattr(self.agent.q, "train"):
            self.agent.q.train()
        if hasattr(self.agent, "q_tgt") and hasattr(self.agent.q_tgt, "eval"):
            # target siempre eval
            self.agent.q_tgt.eval()

    def _agent_eval_mode(self):
        if hasattr(self.agent, "eval_mode"):
            self.agent.eval_mode()
            return
        if hasattr(self.agent, "q") and hasattr(self.agent.q, "eval"):
            self.agent.q.eval()
        if hasattr(self.agent, "q_tgt") and hasattr(self.agent.q_tgt, "eval"):
            self.agent.q_tgt.eval()

    # -------------------------
    # Curriculum helpers
    # -------------------------
    def _recent_sr(self, lvl: int) -> float:
        q = self.recent_by_level.get(int(lvl))
        if not q or len(q) == 0:
            return 0.0
        return float(np.mean(q))

    def _maybe_advance_curriculum(self):
        """Avanza según el SR del nivel BASE (no el mezclado)."""
        if self.curriculum_level >= self.curriculum_max_level:
            return

        base_sr = self._recent_sr(self.curriculum_level)

        if base_sr >= self.advance_threshold:
            if self.mix_next_prob < 0.8:
                self.mix_next_prob = min(0.8, self.mix_next_prob + 0.2)
            else:
                old = self.curriculum_level
                self.curriculum_level += 1
                self.mix_next_prob = 0.0
                self._on_curriculum_advanced(old, self.curriculum_level)

    def _sample_curriculum_level_for_episode(self, rng: np.random.Generator) -> int:
        if self.curriculum_level >= self.curriculum_max_level:
            return self.curriculum_level

        if self.mix_next_prob <= 0.0:
            return self.curriculum_level

        if rng.random() < self.mix_next_prob:
            return min(self.curriculum_level + 1, self.curriculum_max_level)
        return self.curriculum_level

    def _on_curriculum_advanced(self, old_level: int, new_level: int):
        if hasattr(self.agent, "on_curriculum_advanced"):
            self.agent.on_curriculum_advanced(
                old_level=old_level,
                new_level=new_level,
                epsilon_reset_on_advance=self.epsilon_reset_on_advance,
                epsilon_reset_value=self.epsilon_reset_value,
                epsilon_reset_steps=self.epsilon_reset_steps,
                reset_replay_on_advance=self.reset_replay_on_advance,
            )
        else:
            if self.reset_replay_on_advance and hasattr(self.agent, "buffer") and hasattr(self.agent.buffer, "reset"):
                self.agent.buffer.reset()

    # -------------------------
    # Rescue helpers
    # -------------------------
    def _trigger_rescue_boost(self):
        """Dispara un epsilon-boost temporal de forma explícita."""
        if hasattr(self.agent, "trigger_exploration_boost"):
            self.agent.trigger_exploration_boost(
                value=float(self.rescue_epsilon_value),
                steps=int(self.rescue_epsilon_steps),
            )
            return

        # fallback: reutiliza hook existente
        if hasattr(self.agent, "on_curriculum_advanced"):
            self.agent.on_curriculum_advanced(
                old_level=self.curriculum_level,
                new_level=self.curriculum_level,
                epsilon_reset_on_advance=True,
                epsilon_reset_value=float(self.rescue_epsilon_value),
                epsilon_reset_steps=int(self.rescue_epsilon_steps),
                reset_replay_on_advance=False,
            )

    def _maybe_rescue(self, *, base_sr: float, ep: int):
        if not self.rescue_on_stuck:
            return

        if self.curriculum_level != self.rescue_level:
            self._rescue_bad_count = 0
            if self._rescue_cooldown_left > 0:
                self._rescue_cooldown_left -= 1
            return

        if self._rescue_cooldown_left > 0:
            self._rescue_cooldown_left -= 1
            return

        if float(base_sr) < self.rescue_sr_threshold:
            self._rescue_bad_count += 1
        else:
            self._rescue_bad_count = 0

        if self._rescue_bad_count < self.rescue_patience_episodes:
            return

        self._rescue_bad_count = 0
        self._rescue_cooldown_left = self.rescue_cooldown_episodes

        self._trigger_rescue_boost()

        self.history["rescues"].append({
            "episode": int(ep),
            "level": int(self.curriculum_level),
            "base_sr": float(base_sr),
            "eps_value": float(self.rescue_epsilon_value),
            "eps_steps": int(self.rescue_epsilon_steps),
            "cooldown": int(self.rescue_cooldown_episodes),
        })

    # -------------------------
    # IO helpers
    # -------------------------
    def _save_checkpoint(self, folder: str, name: str):
        """Guarda state_dict en CPU (portabilidad)."""
        ensure_dir(folder)
        path = os.path.join(folder, name)

        state = self.agent.q.state_dict()
        state_cpu = {k: v.detach().cpu() for k, v in state.items()}
        torch.save(state_cpu, path)

    def _evaluate(self, *, episodes: int, curriculum_level: int, seed: int, freeze_pool: bool = True) -> Dict[str, Any]:
        """Eval defensiva: pasa freeze_pool si existe."""
        self._agent_eval_mode()
        if self._eval_accepts_freeze_pool:
            return self.eval_ctrl.evaluate(
                episodes=int(episodes),
                curriculum_level=int(curriculum_level),
                seed=int(seed),
                freeze_pool=bool(freeze_pool),
            )
        return self.eval_ctrl.evaluate(
            episodes=int(episodes),
            curriculum_level=int(curriculum_level),
            seed=int(seed),
        )

    # -------------------------
    # Main loop
    # -------------------------
    def train(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)

        run_id = f"{self.run_name}_{timestamp()}_seed{self.seed}"
        run_path = os.path.join(self.checkpoint_dir, run_id)
        ensure_dir(run_path)

        best_eval_sr_main = -1.0
        best_eval_sr_lvl2 = -1.0

        for ep in range(1, self.num_episodes + 1):
            self._agent_train_mode()

            lvl_ep = self._sample_curriculum_level_for_episode(rng)
            obs, _info = self.env.reset(
                curriculum_level=int(lvl_ep),
                seed=int(rng.integers(0, 10_000_000)),
            )

            total_r = 0.0
            loss_val: Optional[float] = None
            done = False
            truncated = False

            for _t in range(self.max_steps):
                a = self.agent.act(obs, deterministic=False)
                next_obs, r, done, truncated, _info = self.env.step(a)

                terminal = bool(done or truncated)
                self.agent.remember(obs, a, r, next_obs, terminal)

                out = self.agent.learn()
                if out.get("loss") is not None:
                    loss_val = float(out["loss"])

                obs = next_obs
                total_r += float(r)

                if terminal:
                    break

            success = 1 if bool(done) else 0
            self.recent_by_level[int(lvl_ep)].append(success)

            self._maybe_advance_curriculum()
            base_sr = self._recent_sr(self.curriculum_level)
            self._maybe_rescue(base_sr=base_sr, ep=ep)

            self.history["episode"].append(int(ep))
            self.history["loss"].append(loss_val)
            self.history["epsilon"].append(float(getattr(self.agent, "epsilon", 0.0)) if hasattr(self.agent, "epsilon") else None)
            self.history["reward"].append(float(total_r))
            self.history["success"].append(int(success))
            self.history["curriculum_level_base"].append(int(self.curriculum_level))
            self.history["curriculum_level_episode"].append(int(lvl_ep))
            self.history["mix_next_prob"].append(float(self.mix_next_prob))
            self.history["recent_sr_base"].append(float(base_sr))

            if ep % self.eval_every_episodes == 0:
                # MAIN eval: si ya estás en max, evalúa max (lvl2)
                eval_level_main = int(
                    self.curriculum_max_level
                    if self.curriculum_level >= self.curriculum_max_level
                    else self.curriculum_level
                )

                eval_seed = int(rng.integers(0, 10_000_000))
                eval_stats_main = self._evaluate(
                    episodes=self.eval_episodes,
                    curriculum_level=eval_level_main,
                    seed=eval_seed,
                    freeze_pool=True,  # señal limpia
                )

                self.history["eval"].append({
                    "episode": int(ep),
                    "eval_level": int(eval_level_main),
                    "eval_seed": int(eval_seed),
                    **eval_stats_main,
                })

                self._save_checkpoint(run_path, "last_model.pth")

                sr_main = float(eval_stats_main.get("success_rate", 0.0))
                if sr_main > best_eval_sr_main:
                    best_eval_sr_main = sr_main
                    self._save_checkpoint(run_path, "best_model.pth")

                # Eval “oficial” lvl2 para best_lvl2
                eval_level2 = int(self.curriculum_max_level)
                eval_stats_lvl2 = None

                if self.curriculum_max_level >= 2:
                    eval_seed2 = int(rng.integers(0, 10_000_000))
                    eval_stats_lvl2 = self._evaluate(
                        episodes=self.eval_episodes,
                        curriculum_level=eval_level2,
                        seed=eval_seed2,
                        freeze_pool=True,
                    )

                    self.history["eval"].append({
                        "episode": int(ep),
                        "eval_level": int(eval_level2),
                        "eval_seed": int(eval_seed2),
                        "tag": "lvl2",
                        **eval_stats_lvl2,
                    })

                    sr_lvl2 = float(eval_stats_lvl2.get("success_rate", 0.0))
                    if sr_lvl2 > best_eval_sr_lvl2:
                        best_eval_sr_lvl2 = sr_lvl2
                        self._save_checkpoint(run_path, "best_model_lvl2.pth")

                payload: Dict[str, Any] = {
                    "run_id": str(run_id),
                    "curriculum_level": int(self.curriculum_level),
                    "mix_next_prob": float(self.mix_next_prob),
                    "best_eval_success_rate": float(best_eval_sr_main),
                    "last_eval": {"eval_level": int(eval_level_main), **eval_stats_main},
                    "best_eval_success_rate_lvl2": float(best_eval_sr_lvl2),
                }
                if eval_stats_lvl2 is not None:
                    payload["last_eval_lvl2"] = {"eval_level": int(eval_level2), **eval_stats_lvl2}

                save_json(os.path.join(self.results_dir, f"{run_id}.json"), payload)

            if ep % 10 == 0:
                print(
                    f"[EP {ep:5d}] lvl_base={self.curriculum_level} lvl_ep={lvl_ep} mix={self.mix_next_prob:.2f} "
                    f"R={total_r:.3f} SR(base_win)={base_sr:.3f} eps={getattr(self.agent,'epsilon',0):.3f} "
                    f"loss={loss_val}"
                )

        save_json(os.path.join(self.results_dir, f"{run_id}_history.json"), self.history)

        return {
            "run_id": run_id,
            "checkpoint_path": run_path,
            "best_eval_success_rate": float(best_eval_sr_main),
            "best_eval_success_rate_lvl2": float(best_eval_sr_lvl2),
        }