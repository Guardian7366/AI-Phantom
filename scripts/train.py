import argparse
import os
from typing import Dict, Any

import torch
import yaml

from utils.seeding import seed_everything
from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from controllers.training_controller import TrainingController


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

    # --- TRAIN ---
    train_section = cfg.get("train", {})
    if not isinstance(train_section, dict):
        raise ValueError("La sección 'train' debe ser un diccionario.")

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
    }
    unknown_train = set(train_section.keys()) - allowed_train_keys
    if unknown_train:
        raise KeyError(
            f"Claves desconocidas en 'train': {sorted(list(unknown_train))}. "
            f"Permitidas: {sorted(list(allowed_train_keys))}"
        )

    base_seed = int(cfg.get("seed", 42))
    train_seed = int(train_section.get("seed", base_seed))
    seed_everything(train_seed)

    # --- ENV ---
    env_section = cfg.get("env", {})
    if not isinstance(env_section, dict):
        raise ValueError("La sección 'env' debe ser un diccionario.")
    env_kwargs = _strict_kwargs(MazeConfig, env_section, "env")
    env_cfg = MazeConfig(**env_kwargs)
    env = MazeEnvironment(env_cfg, rng_seed=train_seed)

    # --- AGENT ---
    agent_section = cfg.get("agent", {})
    if not isinstance(agent_section, dict):
        raise ValueError("La sección 'agent' debe ser un diccionario.")
    agent_kwargs = _strict_kwargs(DQNConfig, agent_section, "agent")

    # ✅ si YAML no define seed del agente, lo sincronizamos con train_seed
    if "seed" not in agent_kwargs:
        agent_kwargs["seed"] = int(train_seed)

    agent_cfg = DQNConfig(**agent_kwargs)
    agent = DQNAgent(agent_cfg, device=device)

    # --- Validaciones rápidas ---
    num_episodes = _require_positive_int("train.num_episodes", train_section.get("num_episodes", 5000))

    max_steps_per_episode = _require_positive_int(
        "train.max_steps_per_episode",
        train_section.get("max_steps_per_episode", env_cfg.max_steps),
    )
    max_steps_per_episode = min(int(max_steps_per_episode), int(env_cfg.max_steps))

    eval_every = _require_positive_int("train.eval_every_episodes", train_section.get("eval_every_episodes", 50))
    eval_episodes = _require_positive_int("train.eval_episodes", train_section.get("eval_episodes", 200))
    success_window = _require_positive_int("train.success_window", train_section.get("success_window", 200))

    curriculum_max_level = int(train_section.get("curriculum_max_level", 2))
    if curriculum_max_level < 0:
        raise ValueError(f"'train.curriculum_max_level' debe ser >= 0. Recibido: {curriculum_max_level}")

    run_name = _require_nonempty_str("train.run_name", train_section.get("run_name", "maze_dqn_rainbowlite"))
    results_dir = _require_nonempty_str("train.results_dir", train_section.get("results_dir", "results/runs"))
    checkpoint_dir = _require_nonempty_str("train.checkpoint_dir", train_section.get("checkpoint_dir", "results/checkpoints"))

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
        curriculum_max_level=curriculum_max_level,
        epsilon_reset_on_advance=bool(train_section.get("epsilon_reset_on_advance", True)),
        epsilon_reset_value=float(train_section.get("epsilon_reset_value", 0.30)),
        epsilon_reset_steps=int(train_section.get("epsilon_reset_steps", 25_000)),
        reset_replay_on_advance=bool(train_section.get("reset_replay_on_advance", False)),
        rescue_on_stuck=bool(train_section.get("rescue_on_stuck", False)),
        rescue_level=int(train_section.get("rescue_level", 2)),
        rescue_sr_threshold=float(train_section.get("rescue_sr_threshold", 0.55)),
        rescue_patience_episodes=int(train_section.get("rescue_patience_episodes", 200)),
        rescue_epsilon_value=float(train_section.get("rescue_epsilon_value", 0.35)),
        rescue_epsilon_steps=int(train_section.get("rescue_epsilon_steps", 30_000)),
        rescue_cooldown_episodes=int(train_section.get("rescue_cooldown_episodes", 300)),
    )

    result = trainer.train()
    print("Done:", result)


if __name__ == "__main__":
    main()