# scripts/train_phase1.py
from __future__ import annotations

import os
import time
from collections import deque

import torch

from ai_phantom.core import (
    select_device,
    set_global_seed,
    save_checkpoint,
    load_checkpoint,
    safe_torch_compile,
    RunLogger,   # ✅
)

from ai_phantom.envs.maze import MazeConfig, MazeEnv
from ai_phantom.agents.ppo import CnnActorCritic, PPOConfig, PPOTrainer, Policy
from ai_phantom.agents.ppo.buffer import RolloutBuffer
from ai_phantom.controllers import EvalController, EvalConfig
from ai_phantom.planners.bfs import bfs_plan, path_to_actions
from ai_phantom.agents.ppo.action_mask import mask_invalid_actions


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
    # episodio "hard" con prob hard_frac
    if hard_frac <= 0.0:
        return float(cur_wp)
    if float(torch.rand(()).item()) < float(hard_frac):
        return float(hard_wp)
    return float(cur_wp)

def teacher_action(env: MazeEnv) -> int:
    path = bfs_plan(env.walls, env.agent, env.goal)
    if path is None or len(path) < 2:
        return 0
    return int(path_to_actions(path)[0])


def would_bump_or_oob(env: MazeEnv, action: int) -> bool:
    ar, ac = env.agent
    dr, dc = env.ACTIONS[int(action)]
    nr, nc = ar + dr, ac + dc
    if nr < 0 or nr >= env.cfg.height or nc < 0 or nc >= env.cfg.width:
        return True
    return bool(env.walls[nr, nc])


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
    return (d1 >= d0)


@torch.no_grad()
def policy_logp_of_action(
    trainer: PPOTrainer,
    obs_t: torch.Tensor,
    action: int,
    enable_mask: bool = True,
) -> float:
    logits, _v = trainer.model(obs_t)
    logits = mask_invalid_actions(obs_t, logits, enable=bool(enable_mask))

    sanitize = getattr(trainer, "_sanitize_logits", None)
    if callable(sanitize):
        logits = sanitize(logits)
    else:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=float("-inf"))
        all_neginf = torch.isneginf(logits).all(dim=-1)
        if bool(all_neginf.any()):
            logits = logits.clone()
            logits[all_neginf] = 0.0

    dist = torch.distributions.Categorical(logits=logits)
    a_t = torch.tensor([int(action)], device=obs_t.device, dtype=torch.long)
    return float(dist.log_prob(a_t).item())


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
    num_wall_stages = 6
    wall_p_start = 0.02
    wall_p_final = 0.18

    # ------------------------------
    # ✅ Hard interleaving (reduce distribution shift)
    # ------------------------------
    hard_wp = float(wall_p_final)     # 0.18
    hard_frac = 0.15                 # 15% de episodios con wp=0.18 (recomendado 0.10–0.20)

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
    print(f"Obs shape: {obs_shape} (C={obs_shape[0]}) | dist_channel={train_env.cfg.include_dist_channel}")

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

    # ------------------------------
    # ✅ Sprint 1-A: alineación dura horizon
    # ------------------------------
    if int(train_env.cfg.max_steps) != int(ppo_cfg.rollout_len):
        print(
            f"⚠️ Horizon mismatch: env.max_steps={train_env.cfg.max_steps} vs rollout_len={ppo_cfg.rollout_len}. "
            "Forzando env.max_steps = rollout_len para consistencia."
        )
        train_env.cfg.max_steps = int(ppo_cfg.rollout_len)
        eval_env.cfg.max_steps = int(ppo_cfg.rollout_len)

    # Schedules suaves (no afectan gate)
    ent_start, ent_end = 0.003, 0.0005
    mh_start, mh_end = 6, 12

    trainer = PPOTrainer(model=model, cfg=ppo_cfg, device=device)

    dummy = torch.zeros((1, *obs_shape), device=device, dtype=torch.float32)
    trainer.model = safe_torch_compile(trainer.model, device=device, example_input=dummy)

    policy = Policy(model=trainer.model, enable_action_mask=True)

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
        # ✅ Solo pesos del modelo; NO el optimizer de otra fase
        load_checkpoint(ckpt0, model=trainer.model, optimizer=None, map_location=device, restore_rng=False)
        print(f"Loaded warm-start checkpoint: {ckpt0}")

    bc_ckpt = "results/checkpoints/bc_phase1.pt"
    has_bc = os.path.exists(bc_ckpt)
    if has_bc:
        load_checkpoint(bc_ckpt, model=trainer.model, optimizer=None, map_location=device, restore_rng=False)
        print(f"Loaded BC warm-start checkpoint: {bc_ckpt}")

    # ------------------------------
    # Training setup
    # ------------------------------
    best_path = "results/checkpoints/best_phase1.pt"
    best_test_sr_final = -1.0  # guardamos "best" por TEST_FINAL (objetivo real)
    best_val_sr_any = -1.0     # solo para debug/telemetría
    
    test_drop_enable = True
    test_drop_margin = 0.08
    test_drop_boost_updates = 30

    # ✅ Anti-promoción insegura
    stage_promote_requires_test = True
    stage_promote_test_min = 0.60
    stage_promote_drop_margin = 0.05

    total_updates = 800
    eval_every = 25

    # Eval sizes (FAST vs TEST)
    eval_eps_val_fast = 90     # gate (rápido)
    eval_eps_test_final = 140  # más robusto (pero aún razonable)

    # ✅ Teacher mix más bajo + muere antes (con BC)
    if not has_bc:
        teacher_mix_start = 0.18
        teacher_mix_decay_updates = 160
    else:
        teacher_mix_start = 0.08
        teacher_mix_decay_updates = 80

    teacher_mix_end = 0.0
    teacher_mix = 0.0

    # ✅ Rescue guard: menos agresivo + decae antes
    enable_rescue_guard = True
    guard_prob_start = 0.50
    guard_prob_end = 0.0
    guard_prob_decay_updates = 120

    # ✅ Sprint 2 (C): límites para que el teacher nunca domine
    teacher_mix_max = 0.25
    guard_prob_max = 0.85

    # contadores para adaptación
    bad_gate_streak = 0

    # Episodios recientes terminados por loop
    loop_term_recent = deque(maxlen=80)

    # ✅ Sprint 2-E: Fail-mode telemetry
    fail_recent = deque(maxlen=160)          # strings
    timeout_recent = deque(maxlen=160)       # 1 si timeout
    bumpheavy_recent = deque(maxlen=160)     # 1 si bump-heavy / other
    loop_reason_recent = deque(maxlen=160)   # 1 si loop terminal

    loop_stagn_recent = deque(maxlen=160)
    loop_osc_recent = deque(maxlen=160)
    loop_short_recent = deque(maxlen=160)

    loop_hits_recent = deque(maxlen=160)

    # ✅ Cap duro: teacher/rescue no puede dominar
    max_teacher_frac_per_rollout = 0.10  # 10% de pasos del rollout como máximo

    # ------------------------------
    # ✅ Sprint 1 - E: Gate por validación (VAL_FAST)
    # ------------------------------
    stage_thresholds = [0.35, 0.55, 0.70, 0.82, 0.92, 0.98]  # len == num_wall_stages
    if len(stage_thresholds) != int(num_wall_stages):
        raise RuntimeError("stage_thresholds debe tener el mismo largo que num_wall_stages.")

    need_k = 2  # evals consecutivos
    good_streak = 0

    rollback_enable = True
    rollback_bad_streak = 10
    rollback_to_prev_stage = True

    # ✅ Bonus temporal al cambiar de stage (evita colapsos)
    stage_transition_boost_updates = 0
    stage_boost_len = 35           # 25–50 suele ir bien
    stage_teacher_boost = 1.6      # multiplica teacher_mix durante boost
    stage_guard_boost = 1.4        # multiplica guard_prob durante boost

    # ------------------------------
    # Runtime stats
    # ------------------------------
    episodes = 0
    successes = 0
    recent = deque(maxlen=200)

    # wall seed por episodio
    episode_wall_counter = 0
    episode_reset_seed_base = 123_456

    # init stage + walls + reset
    cur_stage = 0
    cur_wall_prob = wall_prob_for_stage(cur_stage, num_wall_stages, wall_p_start, wall_p_final)
    train_env.rebuild_walls(seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter, wall_prob=cur_wall_prob)
    init_seed = episode_reset_seed_base + episodes
    obs, info = train_env.reset(seed=int(init_seed), phase=1)
    print(f"🧱 init: stage={cur_stage}/{num_wall_stages-1} wall_prob={cur_wall_prob:.3f}")

    # métricas teacher/rescue globales
    rescue_used_total = 0
    steps_total = 0

    t0 = time.time()

    recovery_steps_left = 0
    recovery_steps = 6   # 4–8 recomendado

    for upd in range(1, int(total_updates) + 1):
        buffer.reset()

        # schedules (no gate)
        prog = (upd - 1) / max(1, (total_updates - 1))
        train_env.cfg.min_manhattan = int(round(linear_schedule(mh_start, mh_end, prog)))
        trainer.cfg.ent_coef = linear_schedule(ent_start, ent_end, prog)

        # ------------------------------
        # ✅ Teacher/Guard adaptativo basado en fail modes
        # ------------------------------
        t_mix = min(1.0, float(upd - 1) / float(max(1, teacher_mix_decay_updates)))
        teacher_mix_base = linear_schedule(teacher_mix_start, teacher_mix_end, t_mix)

        t_guard = min(1.0, float(upd - 1) / float(max(1, guard_prob_decay_updates)))
        guard_prob_base = linear_schedule(guard_prob_start, guard_prob_end, t_guard)

        win_sr = (sum(recent) / len(recent)) if len(recent) > 0 else 0.0

        # rates recientes por fail mode (para la ADAPTACIÓN)
        loop_rate_adapt = (sum(loop_reason_recent) / len(loop_reason_recent)) if len(loop_reason_recent) > 0 else 0.0
        timeout_rate_adapt = (sum(timeout_recent) / len(timeout_recent)) if len(timeout_recent) > 0 else 0.0
        other_rate_adapt = (sum(bumpheavy_recent) / len(bumpheavy_recent)) if len(bumpheavy_recent) > 0 else 0.0

        # Base multiplier por performance (suave, no dominante)
        perf_mult = 1.0
        if win_sr < 0.20:
            perf_mult = 1.6
        elif win_sr < 0.40:
            perf_mult = 1.3
        elif win_sr < 0.60:
            perf_mult = 1.15

        # Gate-streak (suave)
        gate_mult = 1.0
        if bad_gate_streak >= 3:
            gate_mult = 1.2
        if bad_gate_streak >= 6:
            gate_mult = 1.35

        guard_mult = 1.0
        teacher_mult = 1.0

        # loops altos -> subir GUARD fuerte
        if loop_rate_adapt > 0.10:
            guard_mult = max(guard_mult, 1.25)
        if loop_rate_adapt > 0.20:
            guard_mult = max(guard_mult, 1.55)
        if loop_rate_adapt > 0.30:
            guard_mult = max(guard_mult, 1.85)

        # timeouts -> subir TEACHER más
        if timeout_rate_adapt > 0.10:
            teacher_mult = max(teacher_mult, 1.25)
        if timeout_rate_adapt > 0.20:
            teacher_mult = max(teacher_mult, 1.55)
        if timeout_rate_adapt > 0.30:
            teacher_mult = max(teacher_mult, 1.85)

        # other -> subir guard leve
        if other_rate_adapt > 0.20:
            guard_mult = max(guard_mult, 1.15)

        # ✅ Boost temporal post-stage
        if stage_transition_boost_updates > 0:
            teacher_mult = max(teacher_mult, stage_teacher_boost)
            guard_mult = max(guard_mult, stage_guard_boost)
            stage_transition_boost_updates -= 1
            
        # Combinación final (cap)
        teacher_mix = min(
            float(teacher_mix_max),
            float(teacher_mix_base) * float(perf_mult) * float(gate_mult) * float(teacher_mult),
        )
        guard_prob = min(
            float(guard_prob_max),
            float(guard_prob_base) * float(perf_mult) * float(gate_mult) * float(guard_mult),
        )

        # Para logging
        adapt_mult = float(perf_mult) * float(gate_mult)

        rollout_last_done = False

        # cap por rollout
        used_teacher_this_rollout = 0
        max_teacher_steps = int(round(max_teacher_frac_per_rollout * float(ppo_cfg.rollout_len)))

        for _t in range(int(ppo_cfg.rollout_len)):
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()

            out = policy.act(obs_t, deterministic=False)
            a_pol = int(out.action.item())
            logp_pol = float(out.logp.item())
            v = float(out.value.item())

            a_used = a_pol
            logp_used = logp_pol
            used_teacher = False

            teacher_allowed = (used_teacher_this_rollout < max_teacher_steps)

            # ✅ Modo recovery: forzar acciones de escape por unos pasos
            if (recovery_steps_left > 0) and teacher_allowed:
                a_used = teacher_action(train_env)
                logp_used = policy_logp_of_action(
                    trainer, obs_t, a_used, enable_mask=trainer.cfg.enable_action_mask
                )
                used_teacher = True
                used_teacher_this_rollout += 1
                rescue_used_total += 1
                recovery_steps_left -= 1

            else:
                # 1) Teacher mix
                if teacher_allowed and teacher_mix > 0.0:
                    if float(torch.rand((), device=device).item()) < float(teacher_mix):
                        a_used = teacher_action(train_env)
                        logp_used = policy_logp_of_action(
                            trainer, obs_t, a_used, enable_mask=trainer.cfg.enable_action_mask
                        )
                        used_teacher = True
                        used_teacher_this_rollout += 1
                        rescue_used_total += 1

            # 2) Rescue guard (+ loop risk)
            if teacher_allowed and enable_rescue_guard and (not used_teacher):
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
                    used_teacher_this_rollout += 1
                    rescue_used_total += 1

            next_obs, r, done, info = train_env.step(a_used)
            # ✅ Recovery trigger por señales REALES del env
            if (not done) and bool(info.get("looped", False)):
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
                episodes += 1
                reached = bool(info.get("reached", False))
                recent.append(1 if reached else 0)
                if reached:
                    successes += 1

                # fail-mode logging (por episodio)
                looped = bool(info.get("looped", False))
                loop_hits = int(info.get("loop_hits", 0))
                loop_reason = info.get("loop_reason", None)

                loop_hits_recent.append(loop_hits)

                lr = str(loop_reason) if loop_reason is not None else ""
                loop_stagn_recent.append(1 if ("stagnation" in lr) else 0)
                loop_osc_recent.append(1 if ("oscillation" in lr) else 0)
                loop_short_recent.append(1 if ("short_cycle" in lr) else 0)

                ended_by_loop = (not reached) and looped and (loop_hits >= 1)
                loop_term_recent.append(1 if ended_by_loop else 0)

                # timeout si llegó al límite de pasos del env sin llegar
                t_now = int(info.get("t", 0))
                timeout = (not reached) and (t_now >= int(train_env.cfg.max_steps))

                # proxy "other"
                bumpheavy = (not reached) and (not ended_by_loop) and (not timeout)

                timeout_recent.append(1 if timeout else 0)
                loop_reason_recent.append(1 if ended_by_loop else 0)
                bumpheavy_recent.append(1 if bumpheavy else 0)

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

                ep_wp = choose_episode_wall_prob(cur_wall_prob, hard_wp, hard_frac)

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
        )

        metrics = trainer.update(buffer)

        if metrics.get("nan_abort", 0.0) > 0.5:
            for g in trainer.optim.param_groups:
                g["lr"] = float(g["lr"]) * 0.5
            print("⚠️ nan_abort detected -> lowering LR x0.5")

        train_sr = (successes / episodes) if episodes > 0 else 0.0
        win_sr = (sum(recent) / len(recent)) if len(recent) > 0 else 0.0
        rescue_rate = rescue_used_total / max(1, steps_total)

        # ✅ Recalcular rates “current” para logging (no stale)
        loop_rate80 = (sum(loop_term_recent) / len(loop_term_recent)) if len(loop_term_recent) > 0 else 0.0
        timeout_rate = (sum(timeout_recent) / len(timeout_recent)) if len(timeout_recent) > 0 else 0.0
        other_rate = (sum(bumpheavy_recent) / len(bumpheavy_recent)) if len(bumpheavy_recent) > 0 else 0.0

        stagn_rate = (sum(loop_stagn_recent)/len(loop_stagn_recent)) if len(loop_stagn_recent)>0 else 0.0
        osc_rate   = (sum(loop_osc_recent)/len(loop_osc_recent)) if len(loop_osc_recent)>0 else 0.0
        short_rate = (sum(loop_short_recent)/len(loop_short_recent)) if len(loop_short_recent)>0 else 0.0

        avg_loop_hits = (sum(loop_hits_recent)/len(loop_hits_recent)) if len(loop_hits_recent)>0 else 0.0

        print(
            f"[UPD {upd:04d}] ep={episodes:5d} SR={train_sr:.3f} win200={win_sr:.3f} loopRate80={loop_rate80:.3f} "
            f"mh={train_env.cfg.min_manhattan:2d} wp_stage={cur_wall_prob:.3f} wp_ep={train_env.cfg.wall_prob:.3f} stage={cur_stage} "
            f"mix={teacher_mix:.3f} guard={guard_prob:.3f} adapt={adapt_mult:.2f} "
            f"cap={max_teacher_steps:3d}/{ppo_cfg.rollout_len} used={used_teacher_this_rollout:3d} "
            f"rescueRate={rescue_rate:.3f} "
            f"lr={metrics.get('lr', trainer.optim.param_groups[0]['lr']):.2e} "
            f"clip={metrics.get('clip', trainer.cfg.clip_range):.3f} "
            f"kl_ema={metrics.get('kl_ema', 0.0):.5f} "
            f"entCoef={trainer.cfg.ent_coef:.5f} "
            f"pi={metrics['pi_loss']:.4f} vf={metrics['vf_loss']:.4f} ev={metrics['explained_var']:.3f} "
            f"ent={metrics['entropy']:.4f} |KL|={metrics['approx_kl']:.5f} stop={int(metrics['early_stop'])} "
            f"nan={int(metrics.get('nan_abort', 0.0) > 0.5)} "
            f"fail(loop={loop_rate80:.2f},to={timeout_rate:.2f},other={other_rate:.2f})"
            f"loops(st={stagn_rate:.2f},osc={osc_rate:.2f},sh={short_rate:.2f},avgHits={avg_loop_hits:.2f}) "
        )

        logger.log({
            "kind": "train_update",
            "upd": int(upd),
            "episodes": int(episodes),
            "successes": int(successes),
            "train_sr_total": float(train_sr),
            "win_sr_200": float(win_sr),

            "stage": int(cur_stage),
            "wall_prob_cur": float(cur_wall_prob),
            "wall_prob_stage": float(cur_wall_prob),
            "wall_prob_episode": float(train_env.cfg.wall_prob),
            "min_manhattan": int(train_env.cfg.min_manhattan),

            "teacher_mix": float(teacher_mix),
            "guard_prob": float(guard_prob),
            "max_teacher_steps": int(max_teacher_steps),
            "used_teacher_this_rollout": int(used_teacher_this_rollout),
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
        })

        # ------------------------------
        # ✅ Gate: VAL_FAST vs TEST_FINAL
        # ------------------------------
        if upd % int(eval_every) == 0:
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

            test = evaluator.evaluate(
                EvalConfig(
                    episodes=int(eval_eps_test_final),
                    phase=1,
                    seed_base=71_000,
                    deterministic=True,
                    rebuild_walls_each_episode=True,
                    walls_seed_base=WALLS_SEED + 900_000,
                    wall_prob=float(wall_p_final),
                )
            )

            test_b = evaluator.evaluate(
                EvalConfig(
                    episodes=int(eval_eps_test_final),
                    phase=1,
                    seed_base=81_000,
                    deterministic=True,
                    rebuild_walls_each_episode=True,
                    walls_seed_base=WALLS_SEED + 950_000,
                    wall_prob=float(wall_p_final),
                )
            )

            best_val_sr_any = max(best_val_sr_any, float(val["sr"]))

            thresh = float(stage_thresholds[cur_stage])
            pass_gate = (float(val["sr"]) >= thresh)

            if pass_gate:
                good_streak += 1
                bad_gate_streak = 0
            else:
                good_streak = 0
                bad_gate_streak += 1

            # ✅ Rollback por colapso prolongado
            if rollback_enable and (not pass_gate) and (bad_gate_streak >= int(rollback_bad_streak)) and (cur_stage > 0):
                old_stage = cur_stage
                cur_stage -= 1
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

                episode_wall_counter = 0
                train_env.rebuild_walls(
                    seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter,
                    wall_prob=cur_wall_prob,
                )

                reset_seed = episode_reset_seed_base + episodes + 10_000 * cur_stage + 777
                obs, info = train_env.reset(seed=int(reset_seed), phase=1)

                # ✅ boost para recuperar estabilidad
                stage_transition_boost_updates = int(stage_boost_len)

                print(f"🧯 ROLLBACK: stage {old_stage} -> {cur_stage} | wall_prob={cur_wall_prob:.3f}")

            print(
                f"   VAL_FAST(stage={cur_stage} wp={cur_wall_prob:.3f}): SR={val['sr']:.3f} "
                f"avg_steps={val['avg_steps']:.1f} | gate: need SR>={thresh:.2f} streak={good_streak}/{need_k} "
                f"| badGate={bad_gate_streak}"
            )
            print(
                f"   TEST_FINAL(wp={wall_p_final:.3f}): SR={test['sr']:.3f} avg_steps={test['avg_steps']:.1f} "
                f"(best_test={best_test_sr_final:.3f})"
                
            )
            print(
                f"   TEST_B(wp={wall_p_final:.3f}): SR={test_b['sr']:.3f} avg_steps={test_b['avg_steps']:.1f}"
            )

            if test_drop_enable and (best_test_sr_final > 0.0):
                min_test_sr = min(float(test["sr"]), float(test_b["sr"]))
                if min_test_sr > best_test_sr_final:
                    best_test_sr_final = min_test_sr
                    stage_transition_boost_updates = max(stage_transition_boost_updates, int(test_drop_boost_updates))
                    print(f"🛡️ TEST drop detected -> boost teacher/guard for {test_drop_boost_updates} updates")

            logger.log({
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
            })

            logger.log({
                "kind": "test_final",
                "upd": int(upd),
                "wall_prob": float(wall_p_final),
                "episodes": int(eval_eps_test_final),
                "sr": float(test["sr"]),
                "avg_steps": float(test["avg_steps"]),
                "best_test_sr_final": float(best_test_sr_final),
            })

            if good_streak >= int(need_k) and cur_stage < (num_wall_stages - 1):

                can_promote = True

                if stage_promote_requires_test:
                    if float(test["sr"]) < float(stage_promote_test_min):
                        can_promote = False

                    if best_test_sr_final > 0.0 and float(test["sr"]) < (
                        best_test_sr_final - float(stage_promote_drop_margin)
                    ):
                        can_promote = False

                if can_promote:
                    old_stage = cur_stage
                    cur_stage += 1
                    good_streak = 0
                    bad_gate_streak = 0

                    cur_wall_prob = wall_prob_for_stage(
                        cur_stage, num_wall_stages, wall_p_start, wall_p_final
                    )

                    episode_wall_counter = 0
                    train_env.rebuild_walls(
                        seed=WALLS_SEED + 1000 * cur_stage + episode_wall_counter,
                        wall_prob=cur_wall_prob,
                    )

                    reset_seed = episode_reset_seed_base + episodes + 10_000 * cur_stage
                    obs, info = train_env.reset(seed=int(reset_seed), phase=1)

                    print(f"🧱 GATE PASS: stage {old_stage} -> {cur_stage} | new wall_prob={cur_wall_prob:.3f}")

                    stage_transition_boost_updates = int(stage_boost_len)

                else:
                    print("🧱 Gate passed, but TEST_FINAL not ready -> holding stage (anti-regression).")    

            min_test_sr = min(float(test["sr"]), float(test_b["sr"]))

            if min_test_sr > best_test_sr_final:
                best_test_sr_final = min_test_sr
                save_checkpoint(
                    best_path,
                    model=trainer.model,
                    optimizer=trainer.optim,
                    extra={
                        "phase": 1,
                        "obs_shape": obs_shape,

                        # Best metrics (separadas)
                        "best_test_sr_det_multiwalls_final": best_test_sr_final,
                        "best_val_sr_any": best_val_sr_any,

                        # Current curriculum state
                        "cur_stage": int(cur_stage),
                        "cur_wall_prob": float(cur_wall_prob),
                        "good_streak": int(good_streak),
                        "need_k": int(need_k),
                        "stage_thresholds": list(stage_thresholds),
                        "bad_gate_streak": int(bad_gate_streak),

                        # Config snapshots
                        "ppo_cfg": ppo_cfg.__dict__,
                        "env_cfg": env_cfg.__dict__,
                        "walls_seed": int(WALLS_SEED),
                        "num_wall_stages": int(num_wall_stages),
                        "wall_p_start": float(wall_p_start),
                        "wall_p_final": float(wall_p_final),

                        # Teacher/rescue setup
                        "teacher_mix_start": float(teacher_mix_start),
                        "teacher_mix_decay_updates": int(teacher_mix_decay_updates),
                        "guard_prob_start": float(guard_prob_start),
                        "guard_prob_decay_updates": int(guard_prob_decay_updates),
                        "max_teacher_frac_per_rollout": float(max_teacher_frac_per_rollout),
                        "teacher_mix_max": float(teacher_mix_max),
                        "guard_prob_max": float(guard_prob_max),
                        "has_bc": bool(has_bc),
                    },
                    save_rng=True,
                )
                print(f"   ✅ Saved BEST phase1: {best_path} (TEST_FINAL detSR_multiwalls={best_test_sr_final:.3f})")

    dt = time.time() - t0
    print(f"Done. updates_ran={upd} time_sec={dt:.1f}")
    # al final antes de salir
    logger.close()


if __name__ == "__main__":
    main()