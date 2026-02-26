# scripts/pretrain_bc_phase1.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ai_phantom.core import select_device, set_global_seed, save_checkpoint
from ai_phantom.envs.maze import MazeConfig, MazeEnv
from ai_phantom.agents.ppo import CnnActorCritic
from ai_phantom.planners.bfs import bfs_plan, path_to_actions
from ai_phantom.agents.ppo.action_mask import mask_invalid_actions
from ai_phantom.agents.ppo.logits_utils import sanitize_logits_keep_neginf


@dataclass
class BCConfig:
    # Repro
    seed: int = 123

    # Phase
    phase: int = 1

    # Dataset seeds
    train_seed_base: int = 42

    # Walls (alineado a Phase1)
    use_walls: bool = True
    walls_seed_init: int = 777          # seed inicial (MazeEnv ctor)
    rebuild_walls_each_episode: bool = True
    walls_seed_base: int = 777          # base para rebuild_walls
    wall_prob: float = 0.18

    # Dificultad (fase 1)
    min_manhattan: int = 6

    # Dataset streaming (controla tiempo)
    episodes: int = 2500                # resets por epoch
    steps_per_ep: int = 48              # pasos teacher por episodio (dataset)
    max_steps_env: int = 256            # IMPORTANT: max_steps del env (horizon real)

    # Train
    lr: float = 3e-4
    batch_size: int = 256
    grad_clip: float = 0.5

    # epochs = cuántas veces regeneramos datos nuevos
    epochs: int = 2

    # Logging
    log_every_updates: int = 200         # (antes "steps", pero realmente es updates)


def _sync_horizon_env(env: MazeEnv, *, expected_max_steps: int, where: str) -> None:
    """
    Guard rail tipo sync_horizon:
    - Fuerza env.cfg.max_steps a expected_max_steps
    - Verifica que no se desincronice (y grita con mensaje útil)
    """
    exp = int(expected_max_steps)
    if not hasattr(env, "cfg"):
        raise RuntimeError(f"[sync_horizon] MazeEnv sin atributo cfg en {where}")

    # Forzar
    env.cfg.max_steps = exp

    # Verificar
    got = int(getattr(env.cfg, "max_steps", -1))
    if got != exp:
        raise RuntimeError(
            f"[sync_horizon] max_steps desincronizado en {where}: got={got} expected={exp}. "
            f"Revisa reset()/rebuild_walls() o cualquier lugar que reescriba env.cfg.max_steps."
        )


def _teacher_action(env: MazeEnv) -> int:
    """
    Teacher BFS: retorna la 1ra acción óptima desde el estado actual.
    En fase=1 tu reset ya garantiza alcanzabilidad (dist_map[start] != -1).
    """
    path = bfs_plan(env.walls, env.agent, env.goal)
    if path is None or len(path) < 2:
        return 0
    return int(path_to_actions(path)[0])


@torch.no_grad()
def _quick_eval_teacher_acc(
    env: MazeEnv,
    model: torch.nn.Module,
    device: torch.device,
    *,
    phase: int,
    seed_base: int,
    episodes: int = 200,
    steps_per_ep: int = 12,
    rebuild_walls_each_episode: bool = False,
    walls_seed_base: int = 777,
    wall_prob: float = 0.18,
    expected_max_steps: int = 256,
) -> float:
    """
    Eval rápida: compara acción del modelo vs acción BFS en estados muestreados.
    OJO: esto NO es SR, solo “imitation accuracy”.
    """
    model.eval()
    correct = 0
    total = 0

    for ep in range(int(episodes)):
        if rebuild_walls_each_episode and hasattr(env, "rebuild_walls"):
            env.rebuild_walls(seed=int(walls_seed_base + seed_base + ep), wall_prob=float(wall_prob))
            _sync_horizon_env(env, expected_max_steps=expected_max_steps, where="quick_eval:after_rebuild_walls")

        obs, _ = env.reset(seed=int(seed_base + ep), phase=int(phase))
        _sync_horizon_env(env, expected_max_steps=expected_max_steps, where="quick_eval:after_reset")

        for _ in range(int(steps_per_ep)):
            a_star = _teacher_action(env)

            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device).float()
            logits, _ = model(obs_t)

            # action masking + sanitize igual a PPO
            logits = mask_invalid_actions(obs_t, logits, enable=True)
            logits = sanitize_logits_keep_neginf(logits, nan_repl=0.0)
            all_neginf = torch.isneginf(logits).all(dim=-1)
            if bool(all_neginf.any()):
                logits = logits.clone()
                logits[all_neginf] = 0.0

            a_hat = int(torch.argmax(logits, dim=-1).item())

            correct += 1 if (a_hat == a_star) else 0
            total += 1

            obs, _, done, _ = env.step(a_star)
            if done:
                break

    return (correct / total) if total > 0 else 0.0


def setup() -> tuple[MazeConfig, MazeEnv, BCConfig]:
    cfg = BCConfig()

    # ✅ Env alineado con train_phase1
    env_cfg = MazeConfig(
        height=12,
        width=12,

        # paredes
        use_walls=bool(cfg.use_walls),
        wall_prob=float(cfg.wall_prob),

        # IMPORTANT: horizon real (guard rail lo mantendrá)
        max_steps=int(cfg.max_steps_env),
        min_manhattan=int(cfg.min_manhattan),

        # Rewards (no importa tanto para BC, pero alineamos)
        step_penalty=-0.01,
        wall_bump_penalty=-0.02,
        goal_reward=1.0,
        progress_reward=0.03,
        revisit_penalty=0.002,

        # canales (forzados)
        include_goal=True,
        include_visited=True,
        include_step_channel=True,

        # dist channel (forzado)
        include_dist_channel=True,
        dist_invert=True,
        dist_clip=64,

        novelty_beta=0.0,
        progress_reward_clip=0.05,
    )

    env = MazeEnv(env_cfg, seed=int(cfg.walls_seed_init))

    return (env_cfg, env, cfg)


def main(env_cfg, env, cfg, verbose=False) -> iter[tuple[int, int | None, int]]:
    set_global_seed(cfg.seed)

    # Guard rails básicos del dataset/horizon
    if int(cfg.steps_per_ep) <= 0:
        raise ValueError(f"steps_per_ep debe ser > 0, recibido={cfg.steps_per_ep}")
    if int(cfg.max_steps_env) <= 0:
        raise ValueError(f"max_steps_env debe ser > 0, recibido={cfg.max_steps_env}")
    if int(cfg.steps_per_ep) > int(cfg.max_steps_env):
        raise ValueError(
            f"steps_per_ep ({cfg.steps_per_ep}) no puede ser > max_steps_env ({cfg.max_steps_env}). "
            f"Sube max_steps_env o baja steps_per_ep."
        )

    dev_cfg = select_device(device="auto", allow_tf32=True, cudnn_benchmark=True)
    device = dev_cfg.device

    os.makedirs("results/checkpoints", exist_ok=True)

    # En BC el ctor seed fija el primer layout; luego rebuild_walls da diversidad si está habilitado
    _sync_horizon_env(env, expected_max_steps=int(cfg.max_steps_env), where="init")

    # Modelo
    obs0, _ = env.reset(seed=0, phase=int(cfg.phase))
    _sync_horizon_env(env, expected_max_steps=int(cfg.max_steps_env), where="after_first_reset")

    obs_shape = tuple(obs0.shape)
    model = CnnActorCritic(obs_shape=obs_shape, num_actions=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), eps=1e-5)

    if verbose:
        print(
            "BC setup:"
            f" use_walls={cfg.use_walls}"
            f" wall_prob={cfg.wall_prob}"
            f" walls_seed_init={cfg.walls_seed_init}"
            f" episodes/epoch={cfg.episodes}"
            f" steps/ep={cfg.steps_per_ep}"
            f" epochs={cfg.epochs}"
            f" horizon(max_steps)={cfg.max_steps_env}"
        )

    model.train()

    # Buffers batch (CPU) -> enviamos a GPU en cada update
    obs_batch: list[np.ndarray] = []
    act_batch: list[int] = []

    step_counter = 0
    update_counter = 0
    successes = 0
    action = None

    for epc in range(int(cfg.epochs)):
        base = int(cfg.train_seed_base + epc * 1_000_000)  # diversidad por epoch

        running_loss = 0.0
        running_n = 0

        for ep in range(int(cfg.episodes)):
            if cfg.rebuild_walls_each_episode and hasattr(env, "rebuild_walls"):
                ws = int(cfg.walls_seed_base + base + ep)
                env.rebuild_walls(seed=ws, wall_prob=float(cfg.wall_prob))
                _sync_horizon_env(env, expected_max_steps=int(cfg.max_steps_env), where="train:after_rebuild_walls")

            obs, _ = env.reset(seed=int(base + ep), phase=int(cfg.phase))
            _sync_horizon_env(env, expected_max_steps=int(cfg.max_steps_env), where="train:after_reset")

            # Recolecta pasos teacher
            for _ in range(int(cfg.steps_per_ep)):
                action = _teacher_action(env)

                obs_batch.append(obs.copy())
                act_batch.append(int(action))

                obs, _, done, info = env.step(action)
                step_counter += 1
                yield (ep, action, successes)

                # Train step cuando llenamos batch
                if len(act_batch) >= int(cfg.batch_size):
                    X = torch.from_numpy(np.stack(obs_batch, axis=0)).to(device).float()
                    y = torch.tensor(act_batch, device=device, dtype=torch.long)

                    logits, _ = model(X)

                    # ✅ action masking + sanitize igual que PPO
                    logits = mask_invalid_actions(X, logits, enable=True)
                    logits = sanitize_logits_keep_neginf(logits, nan_repl=0.0)
                    all_neginf = torch.isneginf(logits).all(dim=-1)
                    if bool(all_neginf.any()):
                        logits = logits.clone()
                        logits[all_neginf] = 0.0

                    loss = F.cross_entropy(logits, y)

                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                    optim.step()

                    update_counter += 1
                    running_loss += float(loss.item()) * len(act_batch)
                    running_n += len(act_batch)

                    obs_batch.clear()
                    act_batch.clear()

                    if (update_counter % max(1, int(cfg.log_every_updates))) == 0:
                        avg_loss = running_loss / float(max(1, running_n))
                        if verbose:
                            print(f"[BC epc {epc+1}/{cfg.epochs}] updates={update_counter:5d} avg_loss={avg_loss:.4f} steps_seen={step_counter}")

                if done:
                    if info.get("reached", False):
                        successes += 1
                    break

        # Flush batch parcial
        if len(act_batch) > 0:
            X = torch.from_numpy(np.stack(obs_batch, axis=0)).to(device).float()
            y = torch.tensor(act_batch, device=device, dtype=torch.long)

            logits, _v = model(X)
            logits = mask_invalid_actions(X, logits, enable=True)
            logits = sanitize_logits_keep_neginf(logits, nan_repl=0.0)
            all_neginf = torch.isneginf(logits).all(dim=-1)
            if bool(all_neginf.any()):
                logits = logits.clone()
                logits[all_neginf] = 0.0

            loss = F.cross_entropy(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            optim.step()

            update_counter += 1
            running_loss += float(loss.item()) * len(act_batch)
            running_n += len(act_batch)

            obs_batch.clear()
            act_batch.clear()

        avg_loss = running_loss / float(max(1, running_n))

        if verbose:
            acc = _quick_eval_teacher_acc(
                env=env,
                model=model,
                device=device,
                phase=int(cfg.phase),
                seed_base=99_000 + 10_000 * epc,
                episodes=200,
                steps_per_ep=12,
                rebuild_walls_each_episode=bool(cfg.rebuild_walls_each_episode),
                walls_seed_base=int(cfg.walls_seed_base),
                wall_prob=float(cfg.wall_prob),
                expected_max_steps=int(cfg.max_steps_env),
            )
            print(f"[BC epoch {epc+1}/{cfg.epochs}] done. avg_loss={avg_loss:.4f} teacher_acc={acc:.3f}")

    out_path = "results/checkpoints/bc_phase1.pt"
    save_checkpoint(
        out_path,
        model=model,
        optimizer=optim,
        extra={
            "phase": int(cfg.phase),
            "obs_shape": obs_shape,
            "bc_cfg": cfg.__dict__,
            "env_cfg": env_cfg.__dict__,
            "walls_seed_init": int(cfg.walls_seed_init),
        },
        save_rng=True,
    )
    print(f"✅ Saved BC checkpoint: {out_path}")


if __name__ == "__main__":
    env_cfg, env, cfg = setup()
    for _ in main(env_cfg, env, cfg, verbose=True):
        pass
