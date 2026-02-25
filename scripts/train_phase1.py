# scripts/train_phase1.py
from __future__ import annotations

import os
import time
import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # (mantener por compat; puede usarse en helpers)

from ai_phantom.core.horizon import sync_horizon
from ai_phantom.core import (
    select_device,
    set_global_seed,
    save_checkpoint,
    load_checkpoint,
    safe_torch_compile,
    RunLogger,
)

from ai_phantom.envs.maze import MazeConfig, MazeEnv
from ai_phantom.agents.ppo import CnnActorCritic, PPOConfig, PPOTrainer, Policy
from ai_phantom.agents.ppo.buffer import RolloutBuffer
from ai_phantom.controllers import EvalController, EvalConfig
from ai_phantom.planners.bfs import bfs_plan, path_to_actions
from ai_phantom.agents.ppo.action_mask import mask_invalid_actions
from ai_phantom.agents.ppo.logits_utils import sanitize_logits_keep_neginf
from ai_phantom.agents.ppo.logits_utils_extra import fix_all_neginf_rows

def linear_schedule(start: float, end: float, t: float) -> float:
    t = float(max(0.0, min(1.0, t)))
    return (1.0 - t) * float(start) + t * float(end)


def lerp(a: float, b: float, t: float) -> float:
    t = float(max(0.0, min(1.0, t)))
    return (1.0 - t) * float(a) + t * float(b)


def wall_prob_for_stage(stage: int, num_stages: int, p0: float, p1: float) -> float:
    if num_stages <= 1:
        return float(p1)
    t = stage / float(num_stages - 1)
    return float(lerp(p0, p1, t))


def choose_episode_wall_prob(cur_wp: float, hard_wp: float, hard_frac: float) -> float:
    if hard_frac <= 0.0:
        return float(cur_wp)
    if random.random() < float(hard_frac):
        return float(hard_wp)
    return float(cur_wp)


def would_bump_or_oob(env: MazeEnv, action: int) -> bool:
    ar, ac = env.agent
    dr, dc = env.ACTIONS[int(action)]
    nr, nc = ar + dr, ac + dc
    if nr < 0 or nr >= env.cfg.height or nc < 0 or nc >= env.cfg.width:
        return True
    return bool(env.walls[nr, nc])


def teacher_action(env: MazeEnv) -> int:
    path = bfs_plan(env.walls, env.agent, env.goal)
    if path is not None and len(path) >= 2:
        return int(path_to_actions(path)[0])

    # ✅ Fallback: escoger una acción válida para no “empeorar” el rescue
    for a in [0, 1, 2, 3]:
        if not would_bump_or_oob(env, a):
            return int(a)
    return 0


def loop_risk_now(env: MazeEnv) -> bool:
    """
    Señal barata para activar recoveries antes de que el env termine por loop.
    - Si la celda actual ya fue visitada varias veces
    - o si hay oscilación visible en el historial (si existe)
    """
    try:
        ar, ac = env.agent
        if int(env.visited[ar, ac]) >= 3:
            return True
        hist = getattr(env, "_pos_hist", None)
        if hist is not None and len(hist) >= 3:
            p3, p2, p1 = hist[-3], hist[-2], hist[-1]
            if (p1 == p3) and (p2 != p1):
                return True
    except Exception:
        pass
    return False


def no_progress_by_distmap(env: MazeEnv, action: int) -> bool:
    dm = getattr(env, "_dist_map", None)
    if dm is None:
        return False
    ar, ac = env.agent
    dr, dc = env.ACTIONS[int(action)]
    nr, nc = ar + dr, ac + dc
    if nr < 0 or nr >= env.cfg.height or nc < 0 or nc >= env.cfg.width:
        return True
    if env.walls[nr, nc]:
        return True
    d0 = int(dm[ar, ac])
    d1 = int(dm[nr, nc])
    if d0 == -1 or d1 == -1:
        return False
    return d1 >= d0

def _ensure_valid_logits_for_categorical(logits: torch.Tensor) -> torch.Tensor:
    """
    Unifica la protección numérica con Trainer/Policy:
    - preserva -inf (mask)
    - corrige NaN/+inf
    - rescata filas degeneradas (todas -inf / sin finitos) con fallback_action
    """
    logits = sanitize_logits_keep_neginf(logits, nan_repl=0.0)
    logits = fix_all_neginf_rows(logits, fill=0.0, fallback_action=0)
    return logits


@torch.no_grad()
def policy_logp_of_action(
    trainer: PPOTrainer,
    obs_t: torch.Tensor,
    action: int,
    enable_mask: bool = True,
) -> float:
    logits, _v = trainer.model(obs_t)
    logits = mask_invalid_actions(obs_t, logits, enable=bool(enable_mask))
    logits = _ensure_valid_logits_for_categorical(logits)

    dist = torch.distributions.Categorical(logits=logits)
    a_t = torch.tensor([int(action)], device=obs_t.device, dtype=torch.long)

    lp = dist.log_prob(a_t)
    # Blindaje final
    if not torch.isfinite(lp).all():
        return 0.0
    return float(lp.item())

@torch.no_grad()
def policy_value(trainer: PPOTrainer, obs_t: torch.Tensor) -> float:
    _logits, v_t = trainer.model(obs_t)
    if v_t.dim() == 2 and v_t.size(-1) == 1:
        v_t = v_t.squeeze(-1)
    v = v_t.item()
    return float(v) if np.isfinite(v) else 0.0

def main() -> None:
    set_global_seed(123)

    dev_cfg = select_device(device="auto", allow_tf32=True, cudnn_benchmark=True)
    device = dev_cfg.device
    print("Device:", device)

    os.makedirs("results/checkpoints", exist_ok=True)

    logger = RunLogger.create(base_dir="results/runs", run_name="phase1")
    print(f"📝 Logging -> {logger.run_dir}")

    WALLS_SEED = 777

    # ------------------------------
    # Curriculum: stages de wall_prob
    # (Sprint 1: gate por validación)
    # ------------------------------
    num_wall_stages = 8
    wall_p_start = 0.02
    wall_p_final = 0.18

    # ------------------------------
    # ✅ Hard interleaving (reduce distribution shift)
    # ------------------------------
    hard_wp = float(wall_p_final)  # 0.18

    def hard_frac_for_stage(stage: int) -> float:
        # Queremos que el PPO entrene FUERTE en el target (wp=0.180) desde stage 0,
        # porque el test oficial es ahí. Aumentamos hard_frac temprano.
        if stage <= 0:
            return 0.70
        if stage <= 1:
            return 0.75
        if stage <= 3:
            return 0.80
        return 0.85

    env_cfg = MazeConfig(
        height=12,
        width=12,
        use_walls=True,
        wall_prob=float(wall_p_final),  # target final (para TEST)
        max_steps=256,
        min_manhattan=6,
        step_penalty=-0.01,
        wall_bump_penalty=-0.02,
        goal_reward=1.0,
        progress_reward=0.03,
        revisit_penalty=0.002,
        include_goal=True,
        include_visited=True,
        include_step_channel=True,
        novelty_beta=0.0,
        use_progress_shaping=True,
        # ✅ Potential-based shaping (alineado con PPO gamma)
        use_potential_shaping=True,
        potential_gamma=0.99,
        potential_coef=0.05,
        potential_clip=0.10,
        disable_legacy_progress_when_potential=True,
        progress_reward_clip=0.05,
        include_dist_channel=True,
        dist_invert=True,
        dist_clip=64,
        # (Sprint 2) anti-loop robusto
        enable_loop_detection=True,
        loop_window=16,
        loop_cycle_len=4,
        stagnation_steps=10,
        stagnation_penalty=0.02,
        oscillation_penalty=0.03,
        short_cycle_penalty=0.03,
        terminate_on_loop=True,
        loop_terminate_hits=3,
        loop_terminate_extra_penalty=0.10,
        # Legacy anti-loop por visits (apagado)
        loop_terminate_visits=0,
        loop_penalty=0.0,
    )

    train_env = MazeEnv(env_cfg, seed=WALLS_SEED)

    # Eval env: también con walls, pero se re-build por episodio (multi-walls)
    eval_env_cfg = MazeConfig(**env_cfg.__dict__)
    eval_env = MazeEnv(eval_env_cfg, seed=WALLS_SEED)

    obs0, _ = train_env.reset(seed=0, phase=1)
    obs_shape = tuple(obs0.shape)
    print(
        f"Obs shape: {obs_shape} (C={obs_shape[0]}) | dist_channel={train_env.cfg.include_dist_channel}"
    )

    model = CnnActorCritic(obs_shape=obs_shape, num_actions=4)

    # ------------------------------
    # PPO Config
    # ------------------------------
    ppo_cfg = PPOConfig(
        rollout_len=256,
        lr=1.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.18,
        vf_coef=0.5,
        ent_coef=0.003,
        max_grad_norm=0.5,
        ppo_epochs=4,
        minibatch_size=64,
        target_kl=0.035,
        vf_clip_range=0.2,
        enable_action_mask=True,
        abort_on_nan=True,
        nan_logits_replacement=0.0,
        adaptive_kl=True,
        kl_low_mult=0.5,
        kl_high_mult=1.5,
        kl_ema_beta=0.90,
        lr_min=2.5e-5,
        lr_max=1.5e-4,
        clip_min=0.10,
        clip_max=0.22,
        lr_down_factor=0.75,
        lr_up_factor=1.02,
        clip_down_factor=0.90,
        clip_up_factor=1.01,
    )

    sync_horizon([train_env, eval_env], ppo_cfg.rollout_len, name="Phase1")

    # ✅ A2: alinear gamma de shaping potencial con PPO
    g = float(ppo_cfg.gamma)
    env_cfg.potential_gamma = g
    train_env.cfg.potential_gamma = g
    eval_env.cfg.potential_gamma = g

    # Schedules suaves (no afectan gate)
    ent_start, ent_end = 0.003, 0.0005
    mh_start, mh_end = 6, 12

    # ✅ Floors persistentes para finisher
    ent_floor = float(ent_end)
    target_kl_floor = float(ppo_cfg.target_kl)
    clip_floor = float(ppo_cfg.clip_range)

    # ✅ Finisher lock flag (adaptive KL freeze)
    finisher_on = False

    trainer = PPOTrainer(model=model, cfg=ppo_cfg, device=device)

    dummy = torch.zeros((1, *obs_shape), device=device, dtype=torch.float32)
    trainer.model = safe_torch_compile(trainer.model, device=device, example_input=dummy)

    policy = Policy(model=trainer.model, enable_action_mask=True)

    # Safety: re-check horizon once (no cost, prevents silent drift)
    sync_horizon([train_env, eval_env], ppo_cfg.rollout_len, name="Phase1(SAFETY)")

    buffer = RolloutBuffer(
        rollout_len=ppo_cfg.rollout_len,
        obs_shape=obs_shape,
        device=device,
        pin_memory=(device.type == "cuda"),
    )

    evaluator = EvalController(env=eval_env, policy=policy, device=device)

    # ------------------------------
    # Warm-start
    # ------------------------------
    ckpt0 = "results/checkpoints/best_phase0.pt"
    if os.path.exists(ckpt0):
        load_checkpoint(
            ckpt0,
            model=trainer.model,
            optimizer=None,  # ✅ no traer optimizer de otra fase
            map_location=device,
            restore_rng=False,
        )
        print(f"Loaded warm-start checkpoint: {ckpt0}")

    bc_ckpt = "results/checkpoints/bc_phase1.pt"
    has_bc = os.path.exists(bc_ckpt)
    if has_bc:
        load_checkpoint(
            bc_ckpt,
            model=trainer.model,
            optimizer=None,
            map_location=device,
            restore_rng=False,
        )
        print(f"Loaded BC warm-start checkpoint: {bc_ckpt}")

    best_path = "results/checkpoints/best_phase1.pt"

    # ✅ Sprint 2 (C): límites para que el teacher nunca domine
    teacher_mix_max = 0.12
    guard_prob_max = 0.60
    max_teacher_frac_per_rollout = 0.06

    episodes = 0
    episode_reset_seed_base = 123_456

    resumed = False

    # ------------------------------
    # ✅ Resume real desde best_phase1.pt (B1)
    # ------------------------------
    resume_path = best_path
    if os.path.exists(resume_path):
        print(f"🔁 Resuming Phase1 from {resume_path}")

        extra = load_checkpoint(
            resume_path,
            model=trainer.model,
            optimizer=trainer.optim,  # ✅ restaurar optimizer
            map_location=device,
            restore_rng=True,  # ✅ restaurar RNG
        )

        # --- Restaurar curriculum ---
        cur_stage = int(extra.get("cur_stage", 0))
        cur_wall_prob = float(
            extra.get(
                "cur_wall_prob",
                wall_prob_for_stage(cur_stage, num_wall_stages, wall_p_start, wall_p_final),
            )
        )

        good_streak = int(extra.get("good_streak", 0))
        bad_gate_streak = int(extra.get("bad_gate_streak", 0))

        best_test_sr_final = float(extra.get("best_test_sr_det_multiwalls_final", -1.0))

        # --- Restaurar finisher floors ---
        ent_floor = float(extra.get("ent_floor", ent_floor))
        target_kl_floor = float(extra.get("target_kl_floor", target_kl_floor))
        clip_floor = float(extra.get("clip_floor", clip_floor))

        trainer.cfg.early_stop_kl_mult = float(
            extra.get("early_stop_kl_mult", trainer.cfg.early_stop_kl_mult)
        )

        # --- Restaurar caps teacher ---
        teacher_mix_max = float(extra.get("teacher_mix_max", teacher_mix_max))
        guard_prob_max = float(extra.get("guard_prob_max", guard_prob_max))
        max_teacher_frac_per_rollout = float(
            extra.get("max_teacher_frac_per_rollout", max_teacher_frac_per_rollout)
        )

        # --- Rebuild walls según stage ---
        episode_wall_counter = 0
        train_env.rebuild_walls(
            seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter,
            wall_prob=cur_wall_prob,
        )

        reset_seed = episode_reset_seed_base + episodes
        obs, info = train_env.reset(seed=int(reset_seed), phase=1)

        print(
            f"   🔄 Restored stage={cur_stage} "
            f"wall_prob={cur_wall_prob:.3f} "
            f"best_test={best_test_sr_final:.3f}"
        )

        resumed = True

    # ------------------------------
    # Training setup
    # ------------------------------
    if not resumed:
        best_test_sr_final = -1.0  # best por TEST_FINAL
    best_val_sr_any = -1.0  # solo para debug/telemetría

    test_drop_streak = 0
    test_drop_streak_restore = 2

    test_drop_enable = True
    test_drop_margin = 0.08
    test_drop_boost_updates = 30

    # ✅ Anti-promoción insegura
    stage_promote_requires_test = True
    stage_promote_drop_margin = 0.05

    # ✅ Promotion patience (evita quedar clavado si VAL_FAST va bien pero TEST_FINAL va apenas abajo)
    promote_hold = 0              # cuántas evaluaciones seguidas pasamos gate pero no promovimos por TEST
    promote_relax_step = 0.01     # cuánto relajamos por “hold”
    promote_relax_max = 0.05      # máximo relajamiento total (NO más de 5%)

    # ✅ Mismatch boost: si gate pasa pero PPO@0.18 no llega, damos un empujón controlado (sin tocar clip)
    mismatch_stuck = 0              # cuántas eval seguidas estamos bloqueados SOLO por TEST (no drop)
    mismatch_boost_updates = 0       # updates restantes de boost
    mismatch_boost_len = 40          # duración del boost (updates)
    mismatch_ent_floor = 0.0012      # piso temporal de entropía durante boost
    mismatch_lr_max = 1.20e-4        # techo temporal de LR durante boost (sigues teniendo adaptive KL)

    total_updates = 800
    eval_every = 25

    # Eval sizes (FAST vs TEST)
    eval_eps_val_fast = 90
    eval_eps_test_final = 140

    # ✅ Teacher mix más bajo + muere antes (con BC)
    if not has_bc:
        teacher_mix_start = 0.18
        teacher_mix_decay_updates = 160
    else:
        teacher_mix_start = 0.08
        teacher_mix_decay_updates = 60

    teacher_mix_end = 0.0
    teacher_mix = 0.0

    # ✅ Rescue guard: menos agresivo + decae antes
    enable_rescue_guard = True
    guard_prob_start = 0.50
    guard_prob_end = 0.0
    guard_prob_decay_updates = 80

    # contadores para adaptación
    bad_gate_streak = 0 if not resumed else bad_gate_streak

    # Episodios recientes terminados por loop
    loop_term_recent = deque(maxlen=80)

    # ✅ Sprint 2-E: Fail-mode telemetry
    fail_recent = deque(maxlen=160)  # strings
    timeout_recent = deque(maxlen=160)  # 1 si timeout
    bumpheavy_recent = deque(maxlen=160)  # 1 si bump-heavy / other
    loop_reason_recent = deque(maxlen=160)  # 1 si loop terminal
    looped_recent = deque(maxlen=160)  # ✅ loop detectado (no solo terminal)

    loop_stagn_recent = deque(maxlen=160)
    loop_osc_recent = deque(maxlen=160)
    loop_short_recent = deque(maxlen=160)

    loop_hits_recent = deque(maxlen=160)

    # ------------------------------
    # ✅ Sprint 1 - Gate por validación (VAL_FAST)
    # ------------------------------
    stage_thresholds = [0.35, 0.45, 0.55, 0.70, 0.82, 0.92, 0.96, 0.98]  # len == 8
    if len(stage_thresholds) != int(num_wall_stages):
        raise RuntimeError("stage_thresholds debe tener el mismo largo que num_wall_stages.")

    # ✅ requerimiento mínimo de TEST_FINAL para promover (sube con stage)
    stage_test_min = [0.55, 0.58, 0.65, 0.70, 0.82, 0.90, 0.95, 0.97]
    if len(stage_test_min) != int(num_wall_stages):
        raise RuntimeError("stage_test_min debe tener el mismo largo que num_wall_stages.")

    need_k = 2
    good_streak = 0 if not resumed else good_streak

    rollback_enable = True
    rollback_bad_streak = 10

    # ✅ Bonus temporal al cambiar de stage (evita colapsos)
    stage_transition_boost_updates = 0
    stage_boost_len = 35
    stage_teacher_boost = 1.6
    stage_guard_boost = 1.4

    # ------------------------------
    # Runtime stats
    # ------------------------------
    successes = 0
    recent = deque(maxlen=200)
    recent_clean = deque(maxlen=200)

    # wall seed por episodio
    episode_wall_counter = 0

    if not resumed:
        cur_stage = 0
        cur_wall_prob = wall_prob_for_stage(cur_stage, num_wall_stages, wall_p_start, wall_p_final)

        episode_wall_counter = 0
        train_env.rebuild_walls(
            seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter,
            wall_prob=cur_wall_prob,
        )

        init_seed = episode_reset_seed_base + episodes
        obs, info = train_env.reset(seed=int(init_seed), phase=1)

        print(f"🧱 init: stage={cur_stage}/{num_wall_stages-1} wall_prob={cur_wall_prob:.3f}")
    else:
        print(
            f"🧱 resume: stage={cur_stage}/{num_wall_stages-1} wall_prob={cur_wall_prob:.3f} "
            f"best_test={best_test_sr_final:.3f}"
        )

    last_ep_wp = float(cur_wall_prob)
    hard_ep_recent = deque(maxlen=200)  # 1 si episodio fue hard_wp

    # métricas teacher/rescue globales
    rescue_used_total = 0
    steps_total = 0

    # Recovery (teacher-forzado por pocos pasos cuando hay loop-risk/loop-hit)
    recovery_steps_left = 0
    recovery_steps = 4

    t0 = time.time()
    upd = 0  # para que el finally no reviente si hay error temprano

    try:
        for upd in range(1, int(total_updates) + 1):
            buffer.reset()

            # schedules (no gate)
            prog = (upd - 1) / max(1, (total_updates - 1))
            train_env.cfg.min_manhattan = int(round(linear_schedule(mh_start, mh_end, prog)))

            ent_scheduled = linear_schedule(ent_start, ent_end, prog)
            trainer.cfg.ent_coef = max(float(ent_floor), float(ent_scheduled))

            # ✅ Asegura que el finisher SÍ se aplica
            trainer.cfg.target_kl = float(target_kl_floor)

            # clip_floor = mínimo, pero no dejarlo “re-expandirse” demasiado
            trainer.cfg.clip_range = max(float(clip_floor), float(trainer.cfg.clip_range))
            trainer._set_clip_clamped(trainer.cfg.clip_range)

            # ✅ Finisher: congelar adaptive KL (no permitir clip/LR subir)
            trainer.cfg.adaptive_kl = False if finisher_on else True

            # --- Safety caps: si estamos bloqueados por TEST o venimos de drops,
            # evitamos que LR/clip vuelvan al techo y provoquen colapsos.
            recent_drop_now = (test_drop_streak >= 1)

            # ✅ Solo caps duros si hay DROP real (anti-regresión)
            if recent_drop_now:
                trainer.cfg.lr_max = min(float(trainer.cfg.lr_max), 8e-5)
                trainer._set_lr_clamped(trainer._get_lr())

                trainer.cfg.clip_max = min(float(trainer.cfg.clip_max), 0.18)
                trainer.cfg.clip_range = min(float(trainer.cfg.clip_range), float(trainer.cfg.clip_max))
                trainer._set_clip_clamped(trainer.cfg.clip_range)

            # ✅ Boost controlado si estamos en mismatch (gate pasa pero PPO test no llega)
            if mismatch_boost_updates > 0 and (not recent_drop_now):
                # subir exploración y permitir un poco más de LR (sin tocar clip)
                ent_floor = max(float(ent_floor), float(mismatch_ent_floor))
                trainer.cfg.lr_max = max(float(trainer.cfg.lr_max), float(mismatch_lr_max))
                mismatch_boost_updates -= 1

            # ------------------------------
            # ✅ Anti-loop schedule por stage (evita colapsos temprano)
            # ------------------------------
            if cur_stage <= 1:
                train_env.cfg.terminate_on_loop = False
                train_env.cfg.loop_terminate_hits = 999999
            else:
                train_env.cfg.terminate_on_loop = True
                train_env.cfg.loop_terminate_hits = 4 if cur_stage == 2 else 3

            # suavizar penalizaciones en stages bajos
            if cur_stage <= 1:
                train_env.cfg.stagnation_penalty = 0.015
                train_env.cfg.oscillation_penalty = 0.02
                train_env.cfg.short_cycle_penalty = 0.02
            else:
                train_env.cfg.stagnation_penalty = 0.02
                train_env.cfg.oscillation_penalty = 0.03
                train_env.cfg.short_cycle_penalty = 0.03

            # ------------------------------
            # ✅ Teacher/Guard adaptativo basado en fail modes
            # ------------------------------
            t_mix = min(1.0, float(upd - 1) / float(max(1, teacher_mix_decay_updates)))
            teacher_mix_base = linear_schedule(teacher_mix_start, teacher_mix_end, t_mix)

            t_guard = min(1.0, float(upd - 1) / float(max(1, guard_prob_decay_updates)))
            guard_prob_base = linear_schedule(guard_prob_start, guard_prob_end, t_guard)

            win_sr = (sum(recent) / len(recent)) if len(recent) > 0 else 0.0

            loop_rate_adapt = (sum(looped_recent) / len(looped_recent)) if len(looped_recent) > 0 else 0.0
            timeout_rate_adapt = (sum(timeout_recent) / len(timeout_recent)) if len(timeout_recent) > 0 else 0.0
            other_rate_adapt = (sum(bumpheavy_recent) / len(bumpheavy_recent)) if len(bumpheavy_recent) > 0 else 0.0

            perf_mult = 1.0
            if win_sr < 0.20:
                perf_mult = 1.6
            elif win_sr < 0.40:
                perf_mult = 1.3
            elif win_sr < 0.60:
                perf_mult = 1.15

            gate_mult = 1.0
            if bad_gate_streak >= 3:
                gate_mult = 1.2
            if bad_gate_streak >= 6:
                gate_mult = 1.35

            guard_mult = 1.0
            teacher_mult = 1.0

            if loop_rate_adapt > 0.10:
                guard_mult = max(guard_mult, 1.25)
            if loop_rate_adapt > 0.20:
                guard_mult = max(guard_mult, 1.55)
            if loop_rate_adapt > 0.30:
                guard_mult = max(guard_mult, 1.85)

            if timeout_rate_adapt > 0.10:
                teacher_mult = max(teacher_mult, 1.25)
            if timeout_rate_adapt > 0.20:
                teacher_mult = max(teacher_mult, 1.55)
            if timeout_rate_adapt > 0.30:
                teacher_mult = max(teacher_mult, 1.85)

            if other_rate_adapt > 0.20:
                guard_mult = max(guard_mult, 1.15)

            if stage_transition_boost_updates > 0:
                teacher_mult = max(teacher_mult, stage_teacher_boost)
                guard_mult = max(guard_mult, stage_guard_boost)
                stage_transition_boost_updates -= 1

            teacher_mix = min(
                float(teacher_mix_max),
                float(teacher_mix_base) * float(perf_mult) * float(gate_mult) * float(teacher_mult),
            )
            guard_prob = min(
                float(guard_prob_max),
                float(guard_prob_base) * float(perf_mult) * float(gate_mult) * float(guard_mult),
            )

            adapt_mult = float(perf_mult) * float(gate_mult)

            rollout_last_done = False

            # cap por rollout
            max_teacher_steps_total = int(round(max_teacher_frac_per_rollout * float(ppo_cfg.rollout_len)))
            max_teacher_steps_mix = int(round(0.06 * float(ppo_cfg.rollout_len)))

            max_teacher_steps_total = max(0, max_teacher_steps_total)
            max_teacher_steps_mix = min(max_teacher_steps_mix, max_teacher_steps_total)
            max_teacher_steps_guard = max(0, max_teacher_steps_total - max_teacher_steps_mix)

            used_teacher_mix = 0
            used_teacher_guard = 0

            ep_used_rescue = False

            for _t in range(int(ppo_cfg.rollout_len)):
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()

                out = policy.act(obs_t, deterministic=False)
                a_pol = int(out.action.item())
                logp_pol = float(out.logp.item())
                v = float(out.value.item())

                a_used = a_pol
                logp_used = logp_pol
                used_teacher = False

                teacher_allowed_mix = used_teacher_mix < max_teacher_steps_mix
                teacher_allowed_guard = used_teacher_guard < max_teacher_steps_guard

                # ✅ Recovery mode: forzar acciones de escape por unos pasos
                if (recovery_steps_left > 0) and teacher_allowed_guard:
                    a_used = teacher_action(train_env)
                    logp_used = policy_logp_of_action(
                        trainer, obs_t, a_used, enable_mask=trainer.cfg.enable_action_mask
                    )
                    v = policy_value(trainer, obs_t)
                    used_teacher = True
                    ep_used_rescue = True
                    used_teacher_guard += 1
                    rescue_used_total += 1
                    recovery_steps_left -= 1
                else:
                    # 1) Teacher mix
                    if teacher_allowed_mix and teacher_mix > 0.0:
                        if random.random() < float(teacher_mix):
                            a_used = teacher_action(train_env)
                            logp_used = policy_logp_of_action(
                                trainer, obs_t, a_used, enable_mask=trainer.cfg.enable_action_mask
                            )
                            used_teacher = True
                            ep_used_rescue = True
                            used_teacher_mix += 1
                            rescue_used_total += 1

                # 2) Rescue guard (+ loop risk)
                if teacher_allowed_guard and enable_rescue_guard and (not used_teacher):
                    bad = would_bump_or_oob(train_env, a_used)

                    if (not bad) and (guard_prob > 0.0) and loop_risk_now(train_env):
                        if float(torch.rand((), device=device).item()) < float(guard_prob):
                            bad = True

                    if (not bad) and (guard_prob > 0.0) and no_progress_by_distmap(train_env, a_used):
                        if float(torch.rand((), device=device).item()) < float(guard_prob):
                            bad = True

                    if bad:
                        a_used = teacher_action(train_env)
                        logp_used = policy_logp_of_action(
                            trainer, obs_t, a_used, enable_mask=trainer.cfg.enable_action_mask
                        )
                        used_teacher = True
                        ep_used_rescue = True
                        used_teacher_guard += 1
                        rescue_used_total += 1

                next_obs, r, done, info = train_env.step(a_used)

                # ✅ Recovery trigger por señales REALES del env
                if not done:
                    looped = bool(info.get("looped", False))
                    lh = int(info.get("loop_hits", 0))
                    lr = str(info.get("loop_reason", ""))
                    npg = int(info.get("no_progress", 0))

                    trigger = False
                    if looped and ((lh >= 2) or ("stagnation" in lr) or ("short_cycle" in lr)):
                        trigger = True

                    if npg >= int(train_env.cfg.stagnation_steps):
                        trigger = True

                    if trigger and (cur_stage >= 2):
                        recovery_steps_left = max(recovery_steps_left, int(recovery_steps))

                rollout_last_done = bool(done)

                buffer.add(
                    obs=obs,
                    action=a_used,
                    reward=float(r),
                    done=bool(done),
                    value=v,
                    logp=logp_used,
                    is_teacher=bool(used_teacher),
                )

                obs = next_obs
                steps_total += 1

                if done:
                    looped_recent.append(1 if bool(info.get("looped", False)) else 0)
                    episodes += 1

                    recovery_steps_left = 0

                    reached = bool(info.get("reached", False))
                    recent.append(1 if reached else 0)
                    if not ep_used_rescue:
                        recent_clean.append(1 if reached else 0)
                    if reached:
                        successes += 1

                    loop_hits = int(info.get("loop_hits", 0))
                    loop_reason = info.get("loop_reason", None)
                    loop_hits_recent.append(loop_hits)

                    lr_str = str(loop_reason) if loop_reason is not None else ""
                    loop_stagn_recent.append(1 if ("stagnation" in lr_str) else 0)
                    loop_osc_recent.append(1 if ("oscillation" in lr_str) else 0)
                    loop_short_recent.append(1 if ("short_cycle" in lr_str) else 0)

                    reason = str(info.get("term_reason", ""))

                    ended_by_loop = reason == "loop"
                    timeout = reason == "timeout"
                    bumpheavy = reason == "other"

                    timeout_recent.append(1 if timeout else 0)
                    loop_reason_recent.append(1 if ended_by_loop else 0)
                    bumpheavy_recent.append(1 if bumpheavy else 0)
                    loop_term_recent.append(1 if ended_by_loop else 0)

                    if reached:
                        fail_recent.append("reached")
                    elif ended_by_loop:
                        fail_recent.append(f"loop:{loop_reason}" if loop_reason else "loop")
                    elif timeout:
                        fail_recent.append("timeout")
                    else:
                        fail_recent.append("other")

                    # al terminar episodio, cambiamos walls seed (mismo stage + wall_prob actual)
                    episode_wall_counter += 1
                    hard_frac = hard_frac_for_stage(cur_stage)

                    # ✅ Boost por bloqueo REAL: si pasamos gate pero el TEST_FINAL_PPO no llega,
                    # entonces estamos en mismatch y debemos entrenar MÁS en wp=0.180.
                    # promote_hold ya cuenta exactamente ese caso.
                    if cur_stage == 0:
                        if promote_hold >= 1:
                            hard_frac = max(hard_frac, 0.80)
                        if promote_hold >= 2:
                            hard_frac = max(hard_frac, 0.85)
                        if promote_hold >= 3:
                            hard_frac = max(hard_frac, 0.90)

                    # (Opcional) si por alguna razón gate falla repetidamente, también empuja hard,
                    # pero esta ya NO es la señal principal.
                    if cur_stage == 0 and bad_gate_streak >= 3:
                        hard_frac = max(hard_frac, 0.80)
                    if cur_stage == 0 and bad_gate_streak >= 6:
                        hard_frac = max(hard_frac, 0.90)

                    hard_frac = float(min(0.95, max(0.0, hard_frac)))

                    # Si pasamos gate pero el test oficial (PPO@0.18) no está listo,
                    # entonces eliminamos el mismatch: entrenar casi siempre en hard_wp.
                    if cur_stage == 0 and promote_hold >= 1:
                        ep_wp = float(hard_wp)  # 0.18 directo
                    else:
                        ep_wp = choose_episode_wall_prob(cur_wall_prob, hard_wp, hard_frac)

                    last_ep_wp = float(ep_wp)
                    hard_ep_recent.append(1 if abs(float(ep_wp) - float(hard_wp)) < 1e-9 else 0)

                    train_env.rebuild_walls(
                        seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter,
                        wall_prob=float(ep_wp),
                    )

                    reset_seed = episode_reset_seed_base + episodes
                    obs, info = train_env.reset(seed=int(reset_seed), phase=1)

            # ---- Bootstrap ----
            if rollout_last_done:
                last_value, last_done = 0.0, True
            else:
                with torch.no_grad():
                    obs_last = torch.from_numpy(obs).unsqueeze(0).to(device).float()
                    last_value = float(policy.value(obs_last).item())
                last_done = False

            buffer.compute_returns_and_advantages(
                last_value=last_value,
                last_done=last_done,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
                exclude_from_adv_norm=buffer.is_teacher,
            )

            metrics = trainer.update(buffer)

            if metrics.get("nan_abort", 0.0) > 0.5:
                for g in trainer.optim.param_groups:
                    g["lr"] = float(g["lr"]) * 0.5
                print("⚠️ nan_abort detected -> lowering LR x0.5")

            train_sr = (successes / episodes) if episodes > 0 else 0.0
            win_sr = (sum(recent) / len(recent)) if len(recent) > 0 else 0.0
            clean_sr = (sum(recent_clean) / len(recent_clean)) if len(recent_clean) > 0 else 0.0
            rescue_rate = rescue_used_total / max(1, steps_total)

            loop_rate80 = (sum(loop_term_recent) / len(loop_term_recent)) if len(loop_term_recent) > 0 else 0.0
            timeout_rate = (sum(timeout_recent) / len(timeout_recent)) if len(timeout_recent) > 0 else 0.0
            other_rate = (sum(bumpheavy_recent) / len(bumpheavy_recent)) if len(bumpheavy_recent) > 0 else 0.0

            stagn_rate = (sum(loop_stagn_recent) / len(loop_stagn_recent)) if len(loop_stagn_recent) > 0 else 0.0
            osc_rate = (sum(loop_osc_recent) / len(loop_osc_recent)) if len(loop_osc_recent) > 0 else 0.0
            short_rate = (sum(loop_short_recent) / len(loop_short_recent)) if len(loop_short_recent) > 0 else 0.0

            avg_loop_hits = (sum(loop_hits_recent) / len(loop_hits_recent)) if len(loop_hits_recent) > 0 else 0.0

            # wall_prob_episode: más robusto que depender de cfg.wall_prob
            wall_prob_episode = float(last_ep_wp)
            hard_rate200 = (sum(hard_ep_recent) / len(hard_ep_recent)) if len(hard_ep_recent) > 0 else 0.0

            print(
                f"[UPD {upd:04d}] ep={episodes:5d} SR={train_sr:.3f} win200={win_sr:.3f} loopRate80={loop_rate80:.3f} cleanSR={clean_sr:.3f} "
                f"mh={train_env.cfg.min_manhattan:2d} wp_stage={cur_wall_prob:.3f} wp_ep={last_ep_wp:.3f} stage={cur_stage} "
                f"mix={teacher_mix:.3f} guard={guard_prob:.3f} adapt={adapt_mult:.2f} boostLeft={stage_transition_boost_updates:2d} "
                f"capT={max_teacher_steps_total:3d} capM={max_teacher_steps_mix:3d} capG={max_teacher_steps_guard:3d}/{ppo_cfg.rollout_len} "
                f"usedT={used_teacher_mix + used_teacher_guard:3d} (mix={used_teacher_mix:3d},guard={used_teacher_guard:3d}) "
                f"rescueRate={rescue_rate:.3f} "
                f"lr={metrics.get('lr', trainer.optim.param_groups[0]['lr']):.2e} "
                f"clip={metrics.get('clip', trainer.cfg.clip_range):.3f} "
                f"kl_ema={metrics.get('kl_ema', 0.0):.5f} "
                f"entCoef={trainer.cfg.ent_coef:.5f} "
                f"pi={metrics['pi_loss']:.4f} vf={metrics['vf_loss']:.4f} ev={metrics['explained_var']:.3f} "
                f"ent={metrics['entropy']:.4f} |KL|={metrics['approx_kl']:.5f} stop={int(metrics['early_stop'])} "
                f"nan={int(metrics.get('nan_abort', 0.0) > 0.5)} "
                f"fail(loop={loop_rate80:.2f},to={timeout_rate:.2f},other={other_rate:.2f}) "
                f"loops(st={stagn_rate:.2f},osc={osc_rate:.2f},sh={short_rate:.2f},avgHits={avg_loop_hits:.2f})"
                f"hard200={hard_rate200:.2f} "
            )

            logger.log(
                {
                    "kind": "train_update",
                    "upd": int(upd),
                    "episodes": int(episodes),
                    "successes": int(successes),
                    "train_sr_total": float(train_sr),
                    "win_sr_200": float(win_sr),
                    "stage": int(cur_stage),
                    "wall_prob_stage": float(cur_wall_prob),
                    "wall_prob_episode": float(wall_prob_episode),
                    "min_manhattan": int(train_env.cfg.min_manhattan),
                    "teacher_mix": float(teacher_mix),
                    "guard_prob": float(guard_prob),
                    "stage_boost_left": int(stage_transition_boost_updates),
                    "max_teacher_steps_total": int(max_teacher_steps_total),
                    "max_teacher_steps_mix": int(max_teacher_steps_mix),
                    "max_teacher_steps_guard": int(max_teacher_steps_guard),
                    "used_teacher_mix": int(used_teacher_mix),
                    "used_teacher_guard": int(used_teacher_guard),
                    "used_teacher_total": int(used_teacher_mix + used_teacher_guard),
                    "rescue_rate_total": float(rescue_rate),
                    "loop_rate80": float(loop_rate80),
                    "avg_loop_hits_recent": float(avg_loop_hits),
                    "timeout_rate": float(timeout_rate),
                    "other_rate": float(other_rate),
                    "ppo_pi_loss": float(metrics["pi_loss"]),
                    "ppo_vf_loss": float(metrics["vf_loss"]),
                    "ppo_entropy": float(metrics["entropy"]),
                    "ppo_ev": float(metrics["explained_var"]),
                    "ppo_kl_abs": float(metrics["approx_kl"]),
                    "ppo_kl_ema": float(metrics.get("kl_ema", 0.0)),
                    "ppo_lr": float(metrics.get("lr", trainer.optim.param_groups[0]["lr"])),
                    "ppo_clip": float(metrics.get("clip", trainer.cfg.clip_range)),
                    "ppo_early_stop": int(metrics.get("early_stop", 0.0) > 0.5),
                    "ppo_nan_abort": int(metrics.get("nan_abort", 0.0) > 0.5),
                }
            )

            # ------------------------------
            # ✅ Gate: VAL_FAST vs TEST_FINAL
            # ------------------------------
            if upd % int(eval_every) == 0:
                # ✅ Guardar estado RNG (anti-deriva por eval)
                py_state = random.getstate()
                np_state = np.random.get_state()
                torch_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

                val = evaluator.evaluate(
                    EvalConfig(
                        episodes=int(eval_eps_val_fast),
                        phase=1,
                        seed_base=31_000 + 1000 * cur_stage,
                        deterministic=True,
                        rebuild_walls_each_episode=True,
                        walls_seed_base=WALLS_SEED + 200_000 + 10_000 * cur_stage,
                        wall_prob=float(cur_wall_prob),
                    )
                )

                test_ppo = evaluator.evaluate(
                    EvalConfig(
                        episodes=int(eval_eps_test_final),
                        phase=1,
                        seed_base=71_000,
                        deterministic=True,
                        rebuild_walls_each_episode=True,
                        walls_seed_base=WALLS_SEED + 900_000,
                        wall_prob=float(wall_p_final),
                        hybrid=False,  # ✅ PPO puro
                    )
                )

                test_h = evaluator.evaluate(
                    EvalConfig(
                        episodes=int(eval_eps_test_final),
                        phase=1,
                        seed_base=71_000,
                        deterministic=True,
                        rebuild_walls_each_episode=True,
                        walls_seed_base=WALLS_SEED + 900_000,
                        wall_prob=float(wall_p_final),
                        hybrid=True,
                        hybrid_min_conf=0.55,
                    )
                )

                test_b_h = evaluator.evaluate(
                    EvalConfig(
                        episodes=int(eval_eps_test_final),
                        phase=1,
                        seed_base=81_000,
                        deterministic=True,
                        rebuild_walls_each_episode=True,
                        walls_seed_base=WALLS_SEED + 950_000,
                        wall_prob=float(wall_p_final),
                        hybrid=True,
                        hybrid_min_conf=0.55,
                    )
                )

                best_val_sr_any = max(best_val_sr_any, float(val["sr"]))

                thresh = float(stage_thresholds[cur_stage])
                pass_gate = float(val["sr"]) >= thresh

                if pass_gate:
                    good_streak += 1
                    bad_gate_streak = 0
                else:
                    good_streak = 0
                    bad_gate_streak += 1
                    promote_hold = 0  # ✅ si no pasamos gate, no acumulamos paciencia de promoción

                # ✅ Rollback por colapso prolongado
                if rollback_enable and (not pass_gate) and (bad_gate_streak >= int(rollback_bad_streak)) and (cur_stage > 0):
                    old_stage = cur_stage
                    cur_stage -= 1
                    promote_hold = 0  # ✅ reset hold al hacer rollback
                    good_streak = 0
                    bad_gate_streak = 0

                    cur_wall_prob = wall_prob_for_stage(cur_stage, num_wall_stages, wall_p_start, wall_p_final)

                    # ✅ Anti-regresión: restaurar mejor checkpoint conocido
                    if os.path.exists(best_path):
                        load_checkpoint(
                            best_path,
                            model=trainer.model,
                            optimizer=trainer.optim,
                            map_location=device,
                            restore_rng=False,
                        )
                        print(f"🧯 Restored BEST checkpoint after rollback: {best_path}")

                        loop_term_recent.clear()
                        fail_recent.clear()
                        timeout_recent.clear()
                        bumpheavy_recent.clear()
                        loop_reason_recent.clear()
                        looped_recent.clear()
                        loop_stagn_recent.clear()
                        loop_osc_recent.clear()
                        loop_short_recent.clear()
                        loop_hits_recent.clear()

                        recovery_steps_left = 0

                    episode_wall_counter = 0
                    train_env.rebuild_walls(
                        seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter,
                        wall_prob=cur_wall_prob,
                    )

                    reset_seed = episode_reset_seed_base + episodes + 10_000 * cur_stage + 777
                    obs, info = train_env.reset(seed=int(reset_seed), phase=1)

                    stage_transition_boost_updates = int(stage_boost_len)

                    print(f"🧯 ROLLBACK: stage {old_stage} -> {cur_stage} | wall_prob={cur_wall_prob:.3f}")

                print(
                    f"   VAL_FAST(stage={cur_stage} wp={cur_wall_prob:.3f}): SR={val['sr']:.3f} "
                    f"avg_steps={val['avg_steps']:.1f} | gate: need SR>={thresh:.2f} streak={good_streak}/{need_k} "
                    f"| badGate={bad_gate_streak}"
                )
                print(
                    f"   TEST_FINAL_HYBRID(wp={wall_p_final:.3f}): SR={test_h['sr']:.3f} avg_steps={test_h['avg_steps']:.1f} "
                    f"bfs_rate={test_h.get('bfs_rate', 0.0):.3f} "
                    f"bfs_try={test_h.get('bfs_try_steps', 0)} fail={test_h.get('bfs_fail_steps', 0)} "
                    f"(best_test={best_test_sr_final:.3f})"
                )
                print(
                    f"   TEST_B_HYBRID(wp={wall_p_final:.3f}): SR={test_b_h['sr']:.3f} avg_steps={test_b_h['avg_steps']:.1f} "
                    f"bfs_rate={test_b_h.get('bfs_rate', 0.0):.3f} "
                    f"bfs_try={test_b_h.get('bfs_try_steps', 0)} fail={test_b_h.get('bfs_fail_steps', 0)}"
                )
                print(
                    f"   TEST_FINAL_PPO(wp={wall_p_final:.3f}): SR={test_ppo['sr']:.3f} avg_steps={test_ppo['avg_steps']:.1f}"
                )

                # --- Métricas oficiales separadas ---
                min_test_hybrid = min(float(test_h["sr"]), float(test_b_h["sr"]))
                sr_test_ppo = float(test_ppo["sr"])

                hy_bfs = max(
                    float(test_h.get("bfs_rate", 0.0)),
                    float(test_b_h.get("bfs_rate", 0.0)),
                )

                # 🔴 MÉTRICA OFICIAL para BEST/DROP (PPO puro manda)
                min_test_sr_official = sr_test_ppo

                # ------------------------------
                # ✅ Finisher mode (empujar SR→1.0)
                # ------------------------------
                if (sr_test_ppo >= 0.93) and (min_test_hybrid >= 0.95) and (hy_bfs <= 0.10):
                    finisher_on = True
                    ent_floor = max(0.0002, float(ent_floor) * 0.85)
                    target_kl_floor = max(0.020, float(target_kl_floor) * 0.90)
                    clip_floor = max(0.12, float(clip_floor) * 0.95)

                    trainer.cfg.lr_max = min(trainer.cfg.lr_max, 8e-5)
                    trainer._set_lr_clamped(trainer._get_lr())

                    teacher_mix_max = min(float(teacher_mix_max), 0.06)
                    guard_prob_max = min(float(guard_prob_max), 0.35)
                    max_teacher_frac_per_rollout = min(float(max_teacher_frac_per_rollout), 0.06)

                    trainer.cfg.early_stop_kl_mult = 1.20
                else:
                    finisher_on = False
                    trainer.cfg.early_stop_kl_mult = 1.50

                prev_best = float(best_test_sr_final)

                # 1️⃣ Detectar DROP vs best anterior
                if test_drop_enable and (prev_best > 0.0):
                    if min_test_sr_official < (prev_best - float(test_drop_margin)):
                        stage_transition_boost_updates = max(
                            stage_transition_boost_updates,
                            int(test_drop_boost_updates),
                        )
                        test_drop_streak += 1

                        trainer._set_lr_clamped(trainer._get_lr() * 0.75)
                        trainer._set_clip_clamped(float(trainer.cfg.clip_range) * 0.90)

                        print(
                            f"🛡️ TEST DROP real: sr_official={min_test_sr_official:.3f} "
                            f"< prev_best={prev_best:.3f} - {test_drop_margin:.2f} "
                            f"(streak={test_drop_streak})"
                        )
                    else:
                        test_drop_streak = 0

                # 2️⃣ DROP persistente -> RESTORE
                if test_drop_streak >= int(test_drop_streak_restore) and os.path.exists(best_path):
                    load_checkpoint(
                        best_path,
                        model=trainer.model,
                        optimizer=trainer.optim,
                        map_location=device,
                        restore_rng=False,
                    )

                    trainer._set_lr_clamped(trainer._get_lr() * 0.85)
                    trainer._set_clip_clamped(float(trainer.cfg.clip_range) * 0.95)

                    print(f"🧯 Restored BEST due to persistent TEST drop: {best_path}")

                    loop_term_recent.clear()
                    fail_recent.clear()
                    timeout_recent.clear()
                    bumpheavy_recent.clear()
                    loop_reason_recent.clear()
                    looped_recent.clear()
                    loop_stagn_recent.clear()
                    loop_osc_recent.clear()
                    loop_short_recent.clear()
                    loop_hits_recent.clear()

                    recovery_steps_left = 0
                    test_drop_streak = 0

                # 3️⃣ Actualizar BEST si mejora
                if min_test_sr_official > prev_best:
                    best_test_sr_final = float(min_test_sr_official)

                    # ✅ Si estamos mejorando BEST, no necesitamos boost
                    mismatch_boost_updates = 0
                    mismatch_stuck = 0

                    save_checkpoint(
                        best_path,
                        model=trainer.model,
                        optimizer=trainer.optim,
                        extra={
                            "phase": 1,
                            "obs_shape": obs_shape,
                            "best_test_sr_det_multiwalls_final": float(best_test_sr_final),
                            "best_val_sr_any": float(best_val_sr_any),
                            "cur_stage": int(cur_stage),
                            "cur_wall_prob": float(cur_wall_prob),
                            "episodes": int(episodes),
                            "successes": int(successes),
                            "steps_total": int(steps_total),
                            "rescue_used_total": int(rescue_used_total),
                            "episode_wall_counter": int(episode_wall_counter),
                            "good_streak": int(good_streak),
                            "need_k": int(need_k),
                            "stage_thresholds": list(stage_thresholds),
                            "stage_test_min": list(stage_test_min),
                            "bad_gate_streak": int(bad_gate_streak),
                            "walls_seed": int(WALLS_SEED),
                            "num_wall_stages": int(num_wall_stages),
                            "wall_p_start": float(wall_p_start),
                            "wall_p_final": float(wall_p_final),
                            "ppo_cfg": ppo_cfg.__dict__,
                            "env_cfg": env_cfg.__dict__,
                            "teacher_mix_start": float(teacher_mix_start),
                            "teacher_mix_decay_updates": int(teacher_mix_decay_updates),
                            "guard_prob_start": float(guard_prob_start),
                            "guard_prob_decay_updates": int(guard_prob_decay_updates),
                            "teacher_mix_max": float(teacher_mix_max),
                            "guard_prob_max": float(guard_prob_max),
                            "max_teacher_frac_per_rollout": float(max_teacher_frac_per_rollout),
                            "ent_floor": float(ent_floor),
                            "target_kl_floor": float(target_kl_floor),
                            "clip_floor": float(clip_floor),
                            "early_stop_kl_mult": float(trainer.cfg.early_stop_kl_mult),
                            "has_bc": bool(has_bc),
                            "best_test_sr_det_multiwalls_final_ppo": float(best_test_sr_final),
                            "last_test_sr_ppo": float(sr_test_ppo),
                            "last_test_sr_hybrid_min": float(min_test_hybrid),
                            "last_test_hybrid_bfs_rate": float(hy_bfs),
                        },
                        save_rng=True,
                    )

                    print(
                        f"   ✅ Saved BEST phase1: {best_path} "
                        f"(TEST_FINAL detSR_multiwalls PPO={best_test_sr_final:.3f})"
                    )

                logger.log(
                    {
                        "kind": "val_fast",
                        "upd": int(upd),
                        "stage": int(cur_stage),
                        "wall_prob": float(cur_wall_prob),
                        "episodes": int(eval_eps_val_fast),
                        "sr": float(val["sr"]),
                        "avg_steps": float(val["avg_steps"]),
                        "gate_threshold": float(thresh),
                        "pass_gate": int(pass_gate),
                        "good_streak": int(good_streak),
                        "need_k": int(need_k),
                        "bad_gate_streak": int(bad_gate_streak),
                    }
                )

                logger.log(
                    {
                        "kind": "test_final",
                        "upd": int(upd),
                        "wall_prob": float(wall_p_final),
                        "episodes": int(eval_eps_test_final),
                        "sr_hybrid_a": float(test_h["sr"]),
                        "sr_hybrid_b": float(test_b_h["sr"]),
                        "min_test_hybrid": float(min_test_hybrid),
                        "sr_ppo": float(sr_test_ppo),
                        "bfs_rate_hybrid": float(hy_bfs),
                        "best_test_sr_final_official": float(best_test_sr_final),
                    }
                )

                
                # ✅ Promoción: requiere pasar gate y NO romper test (anti-regresión)
                if good_streak >= int(need_k) and cur_stage < (num_wall_stages - 1):
                    can_promote = True
                    blocked_by_test = False

                    if stage_promote_requires_test:
                        next_stage = int(min(cur_stage + 1, num_wall_stages - 1))
                        req_base = float(stage_test_min[next_stage])

                        # ✅ Relax controlado si llevamos evaluaciones seguidas pasando gate pero el TEST aún no llega
                        relax = min(float(promote_relax_max), float(promote_hold) * float(promote_relax_step))
                        req_eff = max(0.0, req_base - relax)

                        # ✅ Regla 1: TEST_FINAL_PPO debe alcanzar el requerimiento (con relax)
                        if sr_test_ppo < req_eff:
                            can_promote = False
                            blocked_by_test = True

                        # ✅ Regla 2: anti-regresión dura vs BEST (no se relaja)
                        if float(best_test_sr_final) > 0.0 and sr_test_ppo < (
                            float(best_test_sr_final) - float(stage_promote_drop_margin)
                        ):
                            can_promote = False
                            blocked_by_test = False  # aquí no es “no listo”, es “retroceso”

                    if can_promote:
                        if stage_promote_requires_test:
                            print(f"   PROMOTE_CHECK: next_req_base={req_base:.2f} req_eff={req_eff:.2f} "
                                f"sr_test_ppo={sr_test_ppo:.3f} hold={promote_hold}")
                        old_stage = cur_stage
                        cur_stage += 1
                        good_streak = 0
                        bad_gate_streak = 0
                        promote_hold = 0  # ✅ reset al promover

                        cur_wall_prob = wall_prob_for_stage(
                            cur_stage, num_wall_stages, wall_p_start, wall_p_final
                        )

                        # --- Restaurar contadores para evitar repetir seeds post-resume ---
                        episodes = int(extra.get("episodes", episodes))
                        successes = int(extra.get("successes", successes))
                        steps_total = int(extra.get("steps_total", steps_total))
                        rescue_used_total = int(extra.get("rescue_used_total", rescue_used_total))
                        episode_wall_counter = int(extra.get("episode_wall_counter", 0))

                        train_env.rebuild_walls(
                            seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter,
                            wall_prob=cur_wall_prob,
                        )

                        reset_seed = episode_reset_seed_base + episodes + 10_000 * cur_stage
                        obs, info = train_env.reset(seed=int(reset_seed), phase=1)

                        print(
                            f"🧱 GATE PASS: stage {old_stage} -> {cur_stage} | new wall_prob={cur_wall_prob:.3f}"
                        )

                        stage_transition_boost_updates = int(stage_boost_len)

                    else:
                        # ✅ si estamos bloqueados solo por TEST (no por drop), acumulamos paciencia
                        if blocked_by_test and pass_gate:
                            promote_hold += 1
                            mismatch_stuck += 1
                        else:
                            promote_hold = 0
                            mismatch_stuck = 0

                        # ✅ Cada N bloqueos seguidos por mismatch: boost temporal (no toca clip, solo ent+LRmax)
                        if mismatch_stuck >= 3:
                            mismatch_boost_updates = max(int(mismatch_boost_updates), int(mismatch_boost_len))
                            mismatch_stuck = 0
                            print(f"🚀 MISMATCH BOOST ON: +ent_floor to {mismatch_ent_floor} and lr_max to {mismatch_lr_max:.2e} "
                                f"for {mismatch_boost_len} updates")

                        print(
                            "🧱 Gate passed, but TEST_FINAL not ready -> holding stage (anti-regression). "
                            f"(promote_hold={promote_hold})"
                        )

                # ---- RESTORE RNG ----
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                if torch.cuda.is_available() and cuda_state is not None:
                    torch.cuda.set_rng_state_all(cuda_state)

    finally:
        dt = time.time() - t0
        print(f"Done. updates_ran={upd} time_sec={dt:.1f}")
        logger.close()


if __name__ == "__main__":
    main()