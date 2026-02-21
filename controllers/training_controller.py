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
    - Eval rápida vs eval pesada (para bajar tiempo sin perder precisión)
    - Eval defensiva: usa freeze_pool si existe
    - Modo train/eval del agente si expone train_mode/eval_mode, y fallback a q.train()/q.eval()
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
        eval_episodes: int = 200,                 # (pesada / “oficial”)
        success_window: int = 200,
        # ---- NUEVO: gating para no avanzar con pocas muestras ----
        min_samples_to_advance: Optional[int] = None,  # default: success_window
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
        # ---- rescate por "vueltas" (si hay eval) ----
        rescue_ratio_threshold: float = 3.0,      # mean_ratio_vs_bfs_start alto => ineficiencia

        # ---- NUEVO (no rompe YAML): eval barata para monitoreo ----
        eval_episodes_quick: Optional[int] = None,  # default: min(200, eval_episodes)
        eval_full_every_evals: int = 3,             # cada N evals rápidas hacemos 1 eval pesada lvl2
        # ---- NUEVO (no rompe YAML): interleave lower levels en lvl2 para evitar colapso ----
        interleave_lower_prob: float = 0.15,   # 0.10–0.25 recomendado
        interleave_lower_min_level: int = 1,   # normalmente 1 (lvl1) cuando estás en lvl2
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

        # Eval rápida: por default 200 o menos, pero nunca > eval_episodes
        if eval_episodes_quick is None:
            self.eval_episodes_quick = int(min(200, self.eval_episodes))
        else:
            self.eval_episodes_quick = int(eval_episodes_quick)
        self.eval_episodes_quick = max(50, min(self.eval_episodes_quick, self.eval_episodes))

        self.eval_full_every_evals = max(1, int(eval_full_every_evals))
        self.interleave_lower_prob = float(interleave_lower_prob)
        self.interleave_lower_min_level = int(interleave_lower_min_level)
        self._eval_tick = 0  # cuenta cuántas evals han ocurrido

        self.success_window = int(success_window)
        if min_samples_to_advance is None:
            self.min_samples_to_advance = int(self.success_window)
        else:
            self.min_samples_to_advance = max(1, int(min_samples_to_advance))
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

        self.rescue_ratio_threshold = float(rescue_ratio_threshold)

        self._rescue_bad_count = 0
        self._rescue_cooldown_left = 0

        self.eval_ctrl = EvaluationController(env, agent)

        ensure_dir(self.results_dir)
        ensure_dir(self.checkpoint_dir)

        self.history: Dict[str, Any] = {
            "episode": [],
            "episode_seed": [],

            "loss": [],
            "epsilon": [],
            "reward": [],
            "success": [],

            "invalid_moves": [],
            "revisit_steps": [],
            "visited_unique": [],

            "curriculum_level_base": [],
            "curriculum_level_episode": [],
            "mix_next_prob": [],
            "recent_sr_base": [],

            "td_abs_mean": [],
            "grad_norm": [],

            "eval": [],
            "rescues": [],
        }

        self.curriculum_level = 0
        self.mix_next_prob = 0.0

        self.recent_by_level = {
            lvl: deque(maxlen=self.success_window)
            for lvl in range(self.curriculum_max_level + 1)
        }

        self._eval_accepts_freeze_pool = self._fn_accepts_kw(self.eval_ctrl.evaluate, "freeze_pool")
        self._env_reset_accepts_freeze_pool = self._fn_accepts_kw(getattr(self.env, "reset", None), "freeze_pool")
        self._env_reset_accepts_reset_pool = self._fn_accepts_kw(getattr(self.env, "reset", None), "reset_pool")

        self._last_eval_main: Optional[Dict[str, Any]] = None
        self._last_eval_lvl2: Optional[Dict[str, Any]] = None

    @staticmethod
    def _fn_accepts_kw(fn, kw: str) -> bool:
        try:
            sig = inspect.signature(fn)
            return kw in sig.parameters
        except Exception:
            return False

    # -------------------------
    # Agent mode helpers
    # -------------------------
    def _agent_train_mode(self):
        if hasattr(self.agent, "train_mode"):
            self.agent.train_mode()
            return
        if hasattr(self.agent, "q") and hasattr(self.agent.q, "train"):
            self.agent.q.train()
        if hasattr(self.agent, "q_tgt") and hasattr(self.agent.q_tgt, "eval"):
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
        # ✅ gating: no confiar en ventanas incompletas
        if len(q) < int(self.min_samples_to_advance):
            return 0.0
        return float(np.mean(q))

    def _recent_sr_raw(self, lvl: int) -> float:
        """
        SR REAL de la ventana actual (aunque esté incompleta).
        Sirve SOLO para monitoreo/prints (no para avanzar curriculum).
        """
        q = self.recent_by_level.get(int(lvl))
        if not q or len(q) == 0:
            return 0.0
        return float(np.mean(q))


    def _maybe_advance_curriculum(self, *, ep: int):
        if self.curriculum_level >= self.curriculum_max_level:
            return

        base_sr = self._recent_sr(self.curriculum_level)

        if base_sr >= self.advance_threshold:
            if self.mix_next_prob < 0.8:
                old_mix = float(self.mix_next_prob)
                self.mix_next_prob = min(0.8, self.mix_next_prob + 0.2)

                # ✅ log de evento (no ruido; solo cuando cambia)
                self.history["eval"].append({
                    "episode": int(self.history["episode"][-1]) if self.history["episode"] else None,
                    "tag": "curriculum_mix_up",
                    "level": int(self.curriculum_level),
                    "old_mix": float(old_mix),
                    "new_mix": float(self.mix_next_prob),
                    "base_sr": float(base_sr),
                })
                print(f"[CURR] mix_up lvl={self.curriculum_level} {old_mix:.2f}->{self.mix_next_prob:.2f} base_sr={base_sr:.3f}")

            else:
                old = int(self.curriculum_level)
                self.curriculum_level += 1
                self.mix_next_prob = 0.0
                self._on_curriculum_advanced(old, self.curriculum_level)

                self.history["eval"].append({
                    "episode": int(self.history["episode"][-1]) if self.history["episode"] else None,
                    "tag": "curriculum_advance",
                    "old_level": int(old),
                    "new_level": int(self.curriculum_level),
                    "base_sr": float(base_sr),
                })
                print(f"[CURR] advance {old}->{self.curriculum_level} base_sr={base_sr:.3f}")

    def _sample_curriculum_level_for_episode(self, rng: np.random.Generator) -> int:
        # Si ya estamos en el máximo (ej. lvl2), intercalamos a veces un nivel menor
        # para evitar colapso/olvido y estabilizar la política (muy útil en lvl2).
        if self.curriculum_level >= self.curriculum_max_level:
            p = float(self.interleave_lower_prob)
            if p > 0.0 and rng.random() < p:
                low = max(0, min(int(self.interleave_lower_min_level), self.curriculum_max_level))
                # si low == max, no tiene efecto
                return int(low)
            return int(self.curriculum_level)
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
        if hasattr(self.agent, "trigger_exploration_boost"):
            self.agent.trigger_exploration_boost(
                value=float(self.rescue_epsilon_value),
                steps=int(self.rescue_epsilon_steps),
            )
            return

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

        # Condición A: SR bajo sostenido (ventana base)
        if float(base_sr) < self.rescue_sr_threshold:
            self._rescue_bad_count += 1
        else:
            self._rescue_bad_count = 0

        # Condición B: ineficiencia alta (loops) detectada en eval reciente
        ratio_flag = False
        last_eval = self._last_eval_lvl2 if self._last_eval_lvl2 is not None else self._last_eval_main
        if isinstance(last_eval, dict):
            ratio = last_eval.get("mean_ratio_vs_bfs_start", None)  # ✅ KEY CORRECTA
            if ratio is not None and np.isfinite(float(ratio)) and float(ratio) >= float(self.rescue_ratio_threshold):
                ratio_flag = True

        # Si detectas loops, aceleras el rescate si ya ibas mal
        if ratio_flag and self._rescue_bad_count > 0:
            self._rescue_bad_count += 1

        if self._rescue_bad_count < self.rescue_patience_episodes:
            return

        self._rescue_bad_count = 0
        self._rescue_cooldown_left = self.rescue_cooldown_episodes

        self._trigger_rescue_boost()

        self.history["rescues"].append({
            "episode": int(ep),
            "level": int(self.curriculum_level),
            "base_sr": float(base_sr),
            "ratio_flag": bool(ratio_flag),
            "eps_value": float(self.rescue_epsilon_value),
            "eps_steps": int(self.rescue_epsilon_steps),
            "cooldown": int(self.rescue_cooldown_episodes),
        })

    # -------------------------
    # IO helpers
    # -------------------------
    def _save_checkpoint(self, folder: str, name: str):
        ensure_dir(folder)
        path = os.path.join(folder, name)

        state = self.agent.q.state_dict()
        state_cpu = {k: v.detach().cpu() for k, v in state.items()}
        torch.save(state_cpu, path)

    def _evaluate(self, *, episodes: int, curriculum_level: int, seed: int, freeze_pool: bool = True) -> Dict[str, Any]:
        self._agent_eval_mode()
        if self._eval_accepts_freeze_pool:
            return self.eval_ctrl.evaluate(
                episodes=int(episodes),
                curriculum_level=int(curriculum_level),
                seed=int(seed),
                freeze_pool=bool(freeze_pool),
                reset_pool_on_eval=False,   # ✅ señal comparable entre evals
            )
        return self.eval_ctrl.evaluate(
            episodes=int(episodes),
            curriculum_level=int(curriculum_level),
            seed=int(seed),
            reset_pool_on_eval=False,
        )

    def _env_reset_train(self, *, curriculum_level: int, seed: int):
        """
        Reset para entrenamiento:
        - freeze_pool=False (diversidad)
        - reset_pool=False explícito
        """
        kwargs = {
            "curriculum_level": int(curriculum_level),
            "seed": int(seed),
        }
        if self._env_reset_accepts_freeze_pool:
            kwargs["freeze_pool"] = False
        if self._env_reset_accepts_reset_pool:
            kwargs["reset_pool"] = False
        return self.env.reset(**kwargs)

    # -------------------------
    # Main loop
    # -------------------------
    def train(self) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)

        # Lock del pool de evaluación lvl2 una sola vez (si el env soporta pool)
        if self.curriculum_max_level >= 2 and self._env_reset_accepts_reset_pool:
            kwargs = {"curriculum_level": 2, "seed": int(self.seed), "reset_pool": True}
            if self._env_reset_accepts_freeze_pool:
                kwargs["freeze_pool"] = True
            try:
                _ = self.env.reset(**kwargs)
            except Exception:
                # defensivo: si algo falla, no bloquea el entrenamiento
                pass

        run_id = f"{self.run_name}_{timestamp()}_seed{self.seed}"
        run_path = os.path.join(self.checkpoint_dir, run_id)
        ensure_dir(run_path)

        best_eval_sr_main = -1.0
        best_eval_sr_lvl2 = -1.0

        for ep in range(1, self.num_episodes + 1):
            self._agent_train_mode()

            lvl_ep = self._sample_curriculum_level_for_episode(rng)
            ep_seed = int(rng.integers(0, 10_000_000))

            obs, _info = self._env_reset_train(curriculum_level=int(lvl_ep), seed=int(ep_seed))

            total_r = 0.0
            loss_val: Optional[float] = None
            last_td_abs_mean: Optional[float] = None
            last_grad_norm: Optional[float] = None
            inv_moves_ep = 0
            revisit_steps_ep = 0
            visited_unique_ep = 0

            done = False
            truncated = False

            for _t in range(self.max_steps):
                a = self.agent.act(obs, deterministic=False)
                next_obs, r, done, truncated, _info = self.env.step(a)
                if isinstance(_info, dict):
                    inv_moves_ep = int(_info.get("invalid_moves", inv_moves_ep))
                    revisit_steps_ep = int(_info.get("revisit_steps", revisit_steps_ep))
                    visited_unique_ep = int(_info.get("visited_unique", visited_unique_ep))
                terminal = bool(done or truncated)
                self.agent.remember(obs, a, r, next_obs, terminal)

                out = self.agent.learn()
                if not isinstance(out, dict):
                    out = {}

                if out.get("loss") is not None:
                    loss_val = float(out["loss"])
                if out.get("td_abs_mean") is not None:
                    last_td_abs_mean = float(out["td_abs_mean"])
                if out.get("grad_norm") is not None:
                    last_grad_norm = float(out["grad_norm"])

                obs = next_obs
                total_r += float(r)

                if terminal:
                    break

            reached_goal = False
            if isinstance(_info, dict):
                reached_goal = bool(_info.get("reached_goal", False))

            # ✅ éxito = llegar a la meta (no depende de “done” del env/wrapper)
            success = 1 if reached_goal else 0

            self.recent_by_level[int(lvl_ep)].append(success)

            self._maybe_advance_curriculum(ep=ep)

            # ✅ SR base:
            # - gated: usado para curriculum/rescate (con min_samples_to_advance)
            # - raw: SOLO para monitoreo (ventana parcial)
            base_sr_gated = self._recent_sr(self.curriculum_level)
            base_sr_raw = self._recent_sr_raw(self.curriculum_level)
            base_sr_n = len(self.recent_by_level.get(int(self.curriculum_level), []))

            sr_for_rescue = base_sr_gated
            if self.curriculum_level == self.rescue_level:
                sr_lvl2_gated = self._recent_sr(self.rescue_level)
                sr_for_rescue = sr_lvl2_gated

            self._maybe_rescue(base_sr=sr_for_rescue, ep=ep)

            self.history["episode"].append(int(ep))
            self.history["episode_seed"].append(int(ep_seed))
            self.history["loss"].append(loss_val)
            self.history["epsilon"].append(float(getattr(self.agent, "epsilon", 0.0)) if hasattr(self.agent, "epsilon") else None)
            self.history["reward"].append(float(total_r))
            self.history["success"].append(int(success))
            self.history["invalid_moves"].append(int(inv_moves_ep))
            self.history["revisit_steps"].append(int(revisit_steps_ep))
            self.history["visited_unique"].append(int(visited_unique_ep))
            self.history["curriculum_level_base"].append(int(self.curriculum_level))
            self.history["curriculum_level_episode"].append(int(lvl_ep))
            self.history["mix_next_prob"].append(float(self.mix_next_prob))

            # ✅ SOLO gated a history (para decisiones/plots estables)
            self.history["recent_sr_base"].append(float(base_sr_gated))

            self.history["td_abs_mean"].append(last_td_abs_mean)
            self.history["grad_norm"].append(last_grad_norm)
            if ep % self.eval_every_episodes == 0:
                self._eval_tick += 1

                # MAIN eval: rápida para monitoreo
                eval_level_main = int(
                    self.curriculum_max_level
                    if self.curriculum_level >= self.curriculum_max_level
                    else self.curriculum_level
                )

                eval_seed = int(rng.integers(0, 10_000_000))
                eval_stats_main = self._evaluate(
                    episodes=self.eval_episodes_quick,
                    curriculum_level=eval_level_main,
                    seed=eval_seed,
                    freeze_pool=True,
                )
                self._last_eval_main = dict(eval_stats_main)

                self.history["eval"].append({
                    "episode": int(ep),
                    "eval_level": int(eval_level_main),
                    "eval_seed": int(eval_seed),
                    "tag": "quick",
                    **eval_stats_main,
                })

                print(
                    f"[EVAL {ep:5d}] lvl={eval_level_main} SR={float(eval_stats_main.get('success_rate', 0.0)):.3f} "
                    f"ratio={float(eval_stats_main.get('mean_ratio_vs_bfs_start', float('nan'))):.2f} "
                    f"inv={float(eval_stats_main.get('mean_invalid_moves', float('nan'))):.2f} "
                    f"rev={float(eval_stats_main.get('mean_revisit_steps', float('nan'))):.2f} "
                    f"uniq={float(eval_stats_main.get('mean_visited_unique', float('nan'))):.2f}"
                )

                # checkpoints baratos
                self._save_checkpoint(run_path, "last_model.pth")

                sr_main = float(eval_stats_main.get("success_rate", 0.0))
                if sr_main > best_eval_sr_main:
                    best_eval_sr_main = sr_main
                    self._save_checkpoint(run_path, "best_model.pth")

                # Eval lvl2 pesada SOLO cada N evals (para decidir best_lvl2 con precisión)
                eval_stats_lvl2 = None
                eval_level2 = int(self.curriculum_max_level)

                do_full = (self._eval_tick % self.eval_full_every_evals == 0)
                if do_full and self.curriculum_max_level >= 2:
                    eval_seed2 = int(rng.integers(0, 10_000_000))
                    eval_stats_lvl2 = self._evaluate(
                        episodes=self.eval_episodes,      # ✅ pesada / oficial
                        curriculum_level=eval_level2,
                        seed=eval_seed2,
                        freeze_pool=True,
                    )
                    self._last_eval_lvl2 = dict(eval_stats_lvl2)

                    self.history["eval"].append({
                        "episode": int(ep),
                        "eval_level": int(eval_level2),
                        "eval_seed": int(eval_seed2),
                        "tag": "full_lvl2",
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
                    "best_eval_success_rate_lvl2": float(best_eval_sr_lvl2),
                    "last_eval_quick": {"eval_level": int(eval_level_main), **eval_stats_main},
                }
                if eval_stats_lvl2 is not None:
                    payload["last_eval_full_lvl2"] = {"eval_level": int(eval_level2), **eval_stats_lvl2}

                save_json(os.path.join(self.results_dir, f"{run_id}.json"), payload)

            if ep % 10 == 0:
                eps_val = float(getattr(self.agent, "epsilon", 0.0)) if hasattr(self.agent, "epsilon") else 0.0

                # Para que quede clarísimo:
                # SR_raw = promedio parcial (monitoreo)
                # SR_gated = 0 hasta juntar min_samples_to_advance (para curriculum)
                print(
                    f"[EP {ep:5d}] lvl_base={self.curriculum_level} lvl_ep={lvl_ep} mix={self.mix_next_prob:.2f} "
                    f"R={total_r:.3f} SR_raw(base,n={base_sr_n})={base_sr_raw:.3f} "
                    f"SR_gated(N={int(self.min_samples_to_advance)})={base_sr_gated:.3f} "
                    f"eps={eps_val:.3f} loss={loss_val} inv={inv_moves_ep} rev={revisit_steps_ep} uniq={visited_unique_ep}"
                )

        save_json(os.path.join(self.results_dir, f"{run_id}_history.json"), self.history)

        return {
            "run_id": run_id,
            "checkpoint_path": run_path,
            "best_eval_success_rate": float(best_eval_sr_main),
            "best_eval_success_rate_lvl2": float(best_eval_sr_lvl2),
        }