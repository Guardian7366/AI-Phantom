# scripts/train_ppo.py
from __future__ import annotations
from ai_phantom.core.horizon import sync_horizon

import os
from dataclasses import dataclass

import torch

from ai_phantom.core import select_device, set_global_seed, save_checkpoint, safe_torch_compile
from ai_phantom.envs.maze import MazeConfig, MazeEnv
from ai_phantom.agents.ppo import CnnActorCritic, PPOConfig, PPOTrainer, Policy
from ai_phantom.agents.ppo.buffer import RolloutBuffer
from ai_phantom.controllers import EvalController, EvalConfig


@dataclass
class StopConfig:
    target_det_sr: float = 1.0
    consecutive_evals: int = 3
    min_updates_before_stop: int = 50


def setup() -> None:
    set_global_seed(123)

    # ✅ C3: novelty OFF + shaping clamp
    env_cfg = MazeConfig(
        height=12,
        width=12,
        use_walls=False,
        max_steps=128,  # ✅ Sprint 1-A: alinear con rollout_len
        min_manhattan=6,
        step_penalty=-0.01,
        wall_bump_penalty=-0.02,
        goal_reward=1.0,
        progress_reward=0.03,
        revisit_penalty=0.002,
        include_visited=True,
        include_step_channel=True,
        include_goal=True,
        novelty_beta=0.0,           # ✅ C3
        progress_reward_clip=0.05,  # ✅ C3
        include_dist_channel=True,
        # ✅ Potential-based shaping (alineado con PPO gamma)
        use_potential_shaping=True,
        potential_gamma=0.99,      # igual a ppo_cfg.gamma
        potential_coef=0.05,
        potential_clip=0.10,
        disable_legacy_progress_when_potential=True,
        dist_invert=True, 
        dist_clip=64,
        enable_loop_detection=False,
        terminate_on_loop=False,
    )

    ppo_cfg = PPOConfig(
        rollout_len=128,
        lr=1.0e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.001,
        max_grad_norm=0.5,
        ppo_epochs=4,
        minibatch_size=64,
        target_kl=0.01,
        enable_action_mask=True,
        abort_on_nan=True,
        nan_logits_replacement=0.0,
        lr_max = 1.5e-4,
        clip_max = 0.22,
        lr_up_factor = 1.02,
        clip_up_factor = 1.01,          
    )

    # ✅ A2: alinear gamma de shaping potencial con PPO
    env_cfg.potential_gamma = float(ppo_cfg.gamma)
    
    env = MazeEnv(env_cfg, seed=0)

    # Pass the maze cfg and env
    return (env_cfg, env, ppo_cfg)


def main(env_cfg, env, ppo_cfg, verbose: bool = False):
    dev_cfg = select_device(device="auto", allow_tf32=True, cudnn_benchmark=True)
    device = dev_cfg.device

    sync_horizon([env], ppo_cfg.rollout_len, name="Phase0")
    obs0, _ = env.reset(seed=0, phase=0)
    obs_shape = tuple(obs0.shape)

    model = CnnActorCritic(obs_shape=obs_shape, num_actions=4)
    trainer = PPOTrainer(model=model, cfg=ppo_cfg, device=device)

    dummy = torch.zeros((1, *obs_shape), device=device, dtype=torch.float32)
    trainer.model = safe_torch_compile(trainer.model, device=device, example_input=dummy)

    policy = Policy(
        model=trainer.model,
        enable_action_mask=True,
        nan_repl=float(ppo_cfg.nan_logits_replacement),
        fallback_action=0,
    )

    buffer = RolloutBuffer(
        rollout_len=ppo_cfg.rollout_len,
        obs_shape=obs_shape,
        device=device,
        pin_memory=(device.type == "cuda"),
    )

    evaluator = EvalController(env=env, policy=policy, device=device)

    os.makedirs("results/checkpoints", exist_ok=True)
    best_sr = -1.0
    best_path = "results/checkpoints/best_phase0.pt"

    total_updates = 300
    phase = 0
    train_seed_base = 42

    eval_every = 25
    eval_episodes = 200

    stop_cfg = StopConfig()
    good_eval_streak = 0

    obs, info = env.reset(seed=train_seed_base, phase=phase)
    episodes = 0
    successes = 0
    action = None

    for update_idx in range(1, int(total_updates) + 1):
        buffer.reset()
        rollout_last_done = False
        for _ in range(int(ppo_cfg.rollout_len)):
            yield (episodes, action, best_sr)

            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            out = policy.act(obs_t, deterministic=False)

            action = int(out.action.item())
            logp = float(out.logp.item())
            value = float(out.value.item())

            next_obs, reward, done, info = env.step(action)

            buffer.add(
                obs=obs,
                action=action,
                reward=float(reward),
                done=bool(done),
                value=value,
                logp=logp,
            )

            obs = next_obs
            rollout_last_done = bool(done)

            if done:
                episodes += 1
                if info.get("reached", False):
                    successes += 1
                obs, info = env.reset(seed=train_seed_base + episodes, phase=phase)

        if rollout_last_done:
            last_value = 0.0
            last_done = True
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
        if verbose:
            print(
                f"[UPD {update_idx:04d}] episodes={episodes:5d} trainSR={train_sr:.3f} "
                f"pi={metrics['pi_loss']:.4f} vf={metrics['vf_loss']:.4f} ev={metrics['explained_var']:.3f} "
                f"ent={metrics['entropy']:.4f} kl={metrics['approx_kl']:.5f} stop={int(metrics['early_stop'])} "
                f"nan={int(metrics.get('nan_abort', 0.0) > 0.5)}"
            )

        if update_idx % int(eval_every) == 0:
            ev_det = evaluator.evaluate(EvalConfig(episodes=int(eval_episodes), phase=phase, deterministic=True))
            if verbose:
                print(f"   EVAL(det): SR={ev_det['sr']:.3f} avg_steps={ev_det['avg_steps']:.1f}")

            if float(ev_det["sr"]) > float(best_sr):
                best_sr = float(ev_det["sr"])
                save_checkpoint(
                    best_path,
                    model=trainer.model,
                    optimizer=trainer.optim,
                    extra={
                        "phase": phase,
                        "obs_shape": obs_shape,
                        "best_eval_sr_det": best_sr,
                        "ppo_cfg": ppo_cfg.__dict__,
                        "env_cfg": env_cfg.__dict__,
                    },
                )
                print(f"   ✅ Saved BEST checkpoint: {best_path} (SR={best_sr:.3f})")

            if float(ev_det["sr"]) >= float(stop_cfg.target_det_sr) - 1e-12:
                good_eval_streak += 1
            else:
                good_eval_streak = 0

            if update_idx >= int(stop_cfg.min_updates_before_stop) and good_eval_streak >= int(stop_cfg.consecutive_evals):
                print(f"   🏁 Early-stop: detSR={ev_det['sr']:.3f} for {good_eval_streak} evals (updates >= {stop_cfg.min_updates_before_stop}).")
                break


if __name__ == "__main__":
    env_cfg, env, ppo_cfg = setup()
    for _ in main(env_cfg, env, ppo_cfg, verbose=True):
        pass
