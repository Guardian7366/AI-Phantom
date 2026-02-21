# scripts/train.py
import argparse
import os
import json
from typing import Dict, Any, Optional

import torch
import yaml

from utils.logging import timestamp
from utils.seeding import seed_everything
from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from controllers.training_controller import TrainingController

# ✅ NUEVO: BC trainer (Paso 3)
try:
    from training.trainers.bc_trainer import BCTrainer, BCConfig
except Exception:
    BCTrainer = None
    BCConfig = None

# Wrapper opcional (no rompe si no lo usas)
try:
    from environments.maze.stochastic_wrapper import StochasticWrapper, StochasticConfig
except Exception:
    StochasticWrapper = None
    StochasticConfig = None


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el config YAML: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("El YAML debe ser un diccionario en la raíz.")
    return data


def _strict_kwargs(dataclass_type, section: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    allowed = set(getattr(dataclass_type, "__dataclass_fields__", {}).keys())
    if not allowed:
        raise ValueError(f"{dataclass_type} no parece ser dataclass.")

    unknown = set(section.keys()) - allowed
    if unknown:
        raise KeyError(
            f"Claves desconocidas en '{section_name}': {sorted(list(unknown))}. "
            f"Permitidas: {sorted(list(allowed))}"
        )
    return dict(section)


def _device_from_string(device_str: str) -> torch.device:
    device_str = (device_str or "auto").lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str in ("cuda", "cpu"):
        return torch.device(device_str)
    raise ValueError("device debe ser: 'auto', 'cuda' o 'cpu'.")


def _require_positive_int(name: str, value: Any) -> int:
    try:
        v = int(value)
    except Exception:
        raise ValueError(f"'{name}' debe ser int. Recibido: {value!r}")
    if v <= 0:
        raise ValueError(f"'{name}' debe ser > 0. Recibido: {v}")
    return v


def _require_nonempty_str(name: str, value: Any) -> str:
    s = str(value) if value is not None else ""
    s = s.strip()
    if not s:
        raise ValueError(f"'{name}' no puede ser vacío.")
    return s


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_run_snapshot(folder: str, *, yaml_path: str, cfg: Dict[str, Any], extra: Dict[str, Any]) -> None:
    _ensure_dir(folder)

    resolved_path = os.path.join(folder, "resolved_config.yaml")
    with open(resolved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    meta_path = os.path.join(folder, "run_meta.json")
    payload = {
        "yaml_path": str(yaml_path),
        **extra,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _apply_torch_determinism(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    det = bool(cfg.get("deterministic", False))
    bench = bool(cfg.get("cudnn_benchmark", not det))

    torch.backends.cudnn.benchmark = bool(bench)
    torch.backends.cudnn.deterministic = bool(det)

    allow_tf32 = bool(cfg.get("allow_tf32", True)) and (device.type == "cuda") and (not det)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass

    if det:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    return {
        "deterministic": bool(det),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "allow_tf32": bool(allow_tf32),
    }


# -------------------------
# ✅ NUEVO: Pretrain BC integrado
# -------------------------
def _validate_pretrain_bc_section(section: Dict[str, Any]) -> None:
    allowed = {
        "enabled",
        "dataset_path",
        "save_path",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "use_amp",
        "num_workers",
        "shuffle",
        "seed",
    }
    unknown = set(section.keys()) - allowed
    if unknown:
        raise KeyError(
            f"Claves desconocidas en 'pretrain_bc': {sorted(list(unknown))}. "
            f"Permitidas: {sorted(list(allowed))}"
        )


def _run_pretrain_bc_if_enabled(
    cfg: Dict[str, Any],
    *,
    agent: DQNAgent,
    device: torch.device,
    snapshot_path: str,
) -> Optional[Dict[str, Any]]:
    section = cfg.get("pretrain_bc", None)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError("La sección 'pretrain_bc' debe ser un diccionario si existe.")

    _validate_pretrain_bc_section(section)

    enabled = bool(section.get("enabled", False))
    if not enabled:
        return None

    if BCTrainer is None or BCConfig is None:
        raise ImportError(
            "pretrain_bc.enabled=true pero no se pudo importar BCTrainer/BCConfig. "
            "Verifica training/trainers/bc_trainer.py"
        )

    dataset_path = _require_nonempty_str("pretrain_bc.dataset_path", section.get("dataset_path", ""))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No existe pretrain_bc.dataset_path: {dataset_path}")

    # save_path: si no lo das, lo metemos en el snapshot
    save_path = section.get("save_path", None)
    if save_path is None or str(save_path).strip() == "":
        save_path = os.path.join(snapshot_path, "pretrained_bc.pth")
    else:
        save_path = str(save_path)

    # Construir config BC
    bc_cfg = BCConfig(
        epochs=int(section.get("epochs", 5)),
        batch_size=int(section.get("batch_size", 256)),
        lr=float(section.get("lr", 1e-4)),
        weight_decay=float(section.get("weight_decay", 0.0)),
        grad_clip_norm=float(section.get("grad_clip_norm", 5.0)),
        use_amp=bool(section.get("use_amp", True)),
        num_workers=int(section.get("num_workers", 0)),
        shuffle=bool(section.get("shuffle", True)),
        seed=int(section.get("seed", cfg.get("seed", 42))),
    )

    # Entrenar BC directamente sobre agent.q
    bc_trainer = BCTrainer(agent.q, bc_cfg, device=device)
    stats = bc_trainer.fit_from_npz(dataset_path, save_path=save_path)

    # ✅ sincronizar target con online después del pretrain
    agent.q_tgt.load_state_dict(agent.q.state_dict())
    agent.q_tgt.eval()

    return {
        "enabled": True,
        "dataset_path": dataset_path,
        "save_path": save_path,
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="configs/maze_train.yaml",
        help="Ruta al YAML de entrenamiento.",
    )
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    device = _device_from_string(cfg.get("device", "auto"))
    print("Device:", device)

    # -------------------------
    # Seeds (Ley 4)
    # -------------------------
    base_seed = int(cfg.get("seed", 42))

    train_section = cfg.get("train", {})
    if not isinstance(train_section, dict):
        raise ValueError("La sección 'train' debe ser un diccionario.")

    train_seed = int(train_section.get("seed", base_seed))
    seed_everything(train_seed)
    print("Train seed:", train_seed)

    # Torch determinism knobs
    torch_flags = _apply_torch_determinism(cfg, device)

    # -------------------------
    # TRAIN SECTION VALIDATION
    # -------------------------
    allowed_train_keys = {
        "run_name",
        "num_episodes",
        "max_steps_per_episode",
        "results_dir",
        "checkpoint_dir",
        "seed",
        "eval_every_episodes",
        "eval_episodes",
        "success_window",
        "advance_threshold",
        "min_samples_to_advance",
        "curriculum_max_level",
        # ---- anti-pozo ----
        "epsilon_reset_on_advance",
        "epsilon_reset_value",
        "epsilon_reset_steps",
        "reset_replay_on_advance",
        # ---- rescate ----
        "rescue_on_stuck",
        "rescue_level",
        "rescue_sr_threshold",
        "rescue_patience_episodes",
        "rescue_epsilon_value",
        "rescue_epsilon_steps",
        "rescue_cooldown_episodes",
        # ---- rescate por vueltas ----
        "rescue_ratio_threshold",
        # ---- eval quick/full ----
        "eval_episodes_quick",
        "eval_full_every_evals",
        # ---- interleave ----
        "interleave_lower_prob",
        "interleave_lower_min_level",
    }
    unknown_train = set(train_section.keys()) - allowed_train_keys
    if unknown_train:
        raise KeyError(
            f"Claves desconocidas en 'train': {sorted(list(unknown_train))}. "
            f"Permitidas: {sorted(list(allowed_train_keys))}"
        )

    # Validaciones rápidas
    num_episodes = _require_positive_int("train.num_episodes", train_section.get("num_episodes", 5000))
    eval_every = _require_positive_int("train.eval_every_episodes", train_section.get("eval_every_episodes", 50))
    eval_episodes = _require_positive_int("train.eval_episodes", train_section.get("eval_episodes", 200))
    success_window = _require_positive_int("train.success_window", train_section.get("success_window", 200))

    if eval_episodes >= 600:
        print(
            f"[WARN] eval_episodes={eval_episodes} es alto. Si notas lentitud, baja a 100-200 "
            f"(freeze_pool=True mantiene señal limpia)."
        )

    eval_full_every_evals = int(train_section.get("eval_full_every_evals", 3))
    if eval_full_every_evals <= 1 and eval_episodes >= 200:
        print(
            f"[WARN] eval_full_every_evals={eval_full_every_evals} con eval_episodes={eval_episodes} "
            f"puede ser lento. Recomendado: 3–5."
        )

    curriculum_max_level = int(train_section.get("curriculum_max_level", 2))
    if curriculum_max_level < 0:
        raise ValueError(f"'train.curriculum_max_level' debe ser >= 0. Recibido: {curriculum_max_level}")

    run_name = _require_nonempty_str("train.run_name", train_section.get("run_name", "maze_dqn_rainbowlite"))
    results_dir = _require_nonempty_str("train.results_dir", train_section.get("results_dir", "results/runs"))
    checkpoint_dir = _require_nonempty_str("train.checkpoint_dir", train_section.get("checkpoint_dir", "results/checkpoints"))

    # -------------------------
    # ENV (Ley 1 / Ley 2)
    # -------------------------
    env_section = cfg.get("env", {})
    if not isinstance(env_section, dict):
        raise ValueError("La sección 'env' debe ser un diccionario.")
    env_kwargs = _strict_kwargs(MazeConfig, env_section, "env")
    env_cfg = MazeConfig(**env_kwargs)

    max_steps_per_episode = _require_positive_int(
        "train.max_steps_per_episode",
        train_section.get("max_steps_per_episode", env_cfg.max_steps),
    )
    if int(max_steps_per_episode) > int(env_cfg.max_steps):
        print(
            f"[WARN] max_steps_per_episode ({max_steps_per_episode}) > env.max_steps ({env_cfg.max_steps}). "
            f"Se recorta a env.max_steps."
        )
    max_steps_per_episode = min(int(max_steps_per_episode), int(env_cfg.max_steps))
    print(f"Max steps per episode (effective): {max_steps_per_episode} (env.max_steps={env_cfg.max_steps})")

    env_seed = int(cfg.get("env_seed", train_seed))
    if bool(cfg.get("deterministic", False)) and env_seed != train_seed:
        print(
            f"[WARN] deterministic=true pero env_seed ({env_seed}) != train_seed ({train_seed}). "
            f"Esto es válido, pero revisa si es intencional."
        )

    env = MazeEnvironment(env_cfg, rng_seed=env_seed)

    # -------------------------
    # STOCHASTIC WRAPPER (opcional)
    # -------------------------
    stoch_section = cfg.get("stoch", None)
    stoch_enabled = False
    stoch_meta = None

    if stoch_section is not None:
        if not isinstance(stoch_section, dict):
            raise ValueError("La sección 'stoch' debe ser un diccionario si existe.")

        stoch_enabled = bool(stoch_section.get("enabled", False))
        if stoch_enabled:
            if StochasticWrapper is None or StochasticConfig is None:
                raise ImportError(
                    "stoch.enabled=true pero no se pudo importar StochasticWrapper/StochasticConfig. "
                    "Verifica environments/maze/stochastic_wrapper.py"
                )

            allowed_stoch_keys = set(getattr(StochasticConfig, "__dataclass_fields__", {}).keys()) | {"enabled", "seed"}
            unknown_stoch = set(stoch_section.keys()) - allowed_stoch_keys
            if unknown_stoch:
                raise KeyError(
                    f"Claves desconocidas en 'stoch': {sorted(list(unknown_stoch))}. "
                    f"Permitidas: {sorted(list(allowed_stoch_keys))}"
                )

            stoch_seed = int(stoch_section.get("seed", train_seed + 1_000_003))
            stoch_cfg_kwargs = {
                k: v for k, v in stoch_section.items()
                if k in getattr(StochasticConfig, "__dataclass_fields__", {})
            }
            stoch_cfg = StochasticConfig(**stoch_cfg_kwargs)

            stoch_meta = {
                "seed": int(stoch_seed),
                "config": dict(stoch_cfg_kwargs),
            }

            env = StochasticWrapper(env, stoch_cfg, seed=stoch_seed)
            print(f"StochasticWrapper: enabled (slip_prob={stoch_cfg.action_slip_prob}, seed={stoch_seed})")

    # -------------------------
    # AGENT (Ley 4: seed explícito)
    # -------------------------
    agent_section = cfg.get("agent", {})
    if not isinstance(agent_section, dict):
        raise ValueError("La sección 'agent' debe ser un diccionario.")
    agent_kwargs = _strict_kwargs(DQNConfig, agent_section, "agent")

    if "seed" not in agent_kwargs:
        agent_kwargs["seed"] = int(train_seed)

    agent_cfg = DQNConfig(**agent_kwargs)
    agent = DQNAgent(agent_cfg, device=device)

    # -------------------------
    # RUN SNAPSHOT (Ley 4)
    # -------------------------
    snapshot_root = str(cfg.get("snapshot_dir", checkpoint_dir))
    snapshot_tag = f"{run_name}_seed{train_seed}_{timestamp()}"
    snapshot_path = os.path.join(snapshot_root, snapshot_tag)
    _ensure_dir(snapshot_path)

    # ✅ Pretrain BC antes de guardar meta final (para registrar ruta/estadísticas)
    pretrain_bc_meta = _run_pretrain_bc_if_enabled(
        cfg,
        agent=agent,
        device=device,
        snapshot_path=snapshot_path,
    )

    extra_meta = {
        "device": str(device),
        "train_seed": int(train_seed),
        "env_seed": int(env_seed),
        "agent_seed": int(agent_cfg.seed),
        "stoch_enabled": bool(stoch_enabled),
        "stoch_meta": stoch_meta,
        "torch_flags": dict(torch_flags),
        "pretrain_bc": pretrain_bc_meta,
        "notes": "resolved_config.yaml + run_meta.json permiten reproducir el run; trainer guarda checkpoints y history en results/*",
    }
    _save_run_snapshot(snapshot_path, yaml_path=args.config, cfg=cfg, extra=extra_meta)

    # -------------------------
    # TRAINER
    # -------------------------
    trainer = TrainingController(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        run_name=run_name,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        seed=train_seed,
        eval_every_episodes=eval_every,
        eval_episodes=eval_episodes,
        success_window=success_window,
        advance_threshold=float(train_section.get("advance_threshold", 0.98)),
        min_samples_to_advance=train_section.get("min_samples_to_advance", None),
        curriculum_max_level=curriculum_max_level,
        epsilon_reset_on_advance=bool(train_section.get("epsilon_reset_on_advance", True)),
        epsilon_reset_value=float(train_section.get("epsilon_reset_value", 0.30)),
        epsilon_reset_steps=int(train_section.get("epsilon_reset_steps", 25_000)),
        reset_replay_on_advance=bool(train_section.get("reset_replay_on_advance", False)),
        rescue_on_stuck=bool(train_section.get("rescue_on_stuck", True)),
        rescue_level=int(train_section.get("rescue_level", 2)),
        rescue_sr_threshold=float(train_section.get("rescue_sr_threshold", 0.55)),
        rescue_patience_episodes=int(train_section.get("rescue_patience_episodes", 200)),
        rescue_epsilon_value=float(train_section.get("rescue_epsilon_value", 0.35)),
        rescue_epsilon_steps=int(train_section.get("rescue_epsilon_steps", 30_000)),
        rescue_cooldown_episodes=int(train_section.get("rescue_cooldown_episodes", 300)),
        rescue_ratio_threshold=float(train_section.get("rescue_ratio_threshold", 3.0)),
        eval_episodes_quick=train_section.get("eval_episodes_quick", None),
        eval_full_every_evals=eval_full_every_evals,
        interleave_lower_prob=float(train_section.get("interleave_lower_prob", 0.15)),
        interleave_lower_min_level=int(train_section.get("interleave_lower_min_level", 1)),
    )

    # -------------------------
    # TRAIN
    # -------------------------
    result = trainer.train()
    print("Done:", result)


if __name__ == "__main__":
    main()