# scripts/train_ppo.py
from __future__ import annotations

import argparse
import os
import json
from typing import Dict, Any

import torch
import yaml

from utils.logging import timestamp
from utils.seeding import seed_everything
from environments.maze.maze_env import MazeEnvironment, MazeConfig

from agents.ppo.ppo_agent import PPOAgent, PPOConfig
from training.trainers.ppo_trainer import PPOTrainer, PPOTrainConfig


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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_run_snapshot(folder: str, *, yaml_path: str, cfg: Dict[str, Any], extra: Dict[str, Any]) -> None:
    _ensure_dir(folder)

    resolved_path = os.path.join(folder, "resolved_config.yaml")
    with open(resolved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    meta_path = os.path.join(folder, "run_meta.json")
    payload = {"yaml_path": str(yaml_path), **extra}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/maze_train_ppo.yaml")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    device = _device_from_string(cfg.get("device", "auto"))
    print("Device:", device)

    # Seeds
    base_seed = int(cfg.get("seed", 42))
    seed_everything(base_seed)
    print("Train seed:", base_seed)

    torch_flags = _apply_torch_determinism(cfg, device)

    # ENV
    env_section = cfg.get("env", {}) or {}
    if not isinstance(env_section, dict):
        raise ValueError("La sección 'env' debe ser un diccionario.")
    env_kwargs = _strict_kwargs(MazeConfig, env_section, "env")
    env_cfg = MazeConfig(**env_kwargs)

    env_seed = int(cfg.get("env_seed", base_seed))
    env = MazeEnvironment(env_cfg, rng_seed=env_seed)

    # STOCH wrapper opcional
    stoch_section = cfg.get("stoch", None)
    stoch_enabled = False
    stoch_meta = None

    if stoch_section is not None:
        if not isinstance(stoch_section, dict):
            raise ValueError("La sección 'stoch' debe ser un diccionario si existe.")
        stoch_enabled = bool(stoch_section.get("enabled", False))
        if stoch_enabled:
            if StochasticWrapper is None or StochasticConfig is None:
                raise ImportError("stoch.enabled=true pero no se pudo importar StochasticWrapper/StochasticConfig.")
            allowed_stoch_keys = set(getattr(StochasticConfig, "__dataclass_fields__", {}).keys()) | {"enabled", "seed"}
            unknown = set(stoch_section.keys()) - allowed_stoch_keys
            if unknown:
                raise KeyError(f"Claves desconocidas en 'stoch': {sorted(list(unknown))}.")

            stoch_seed = int(stoch_section.get("seed", base_seed + 1_000_003))
            stoch_cfg_kwargs = {
                k: v for k, v in stoch_section.items()
                if k in getattr(StochasticConfig, "__dataclass_fields__", {})
            }
            stoch_cfg = StochasticConfig(**stoch_cfg_kwargs)

            stoch_meta = {"seed": int(stoch_seed), "config": dict(stoch_cfg_kwargs)}
            env = StochasticWrapper(env, stoch_cfg, seed=stoch_seed)
            print(f"StochasticWrapper: enabled (slip_prob={stoch_cfg.action_slip_prob}, seed={stoch_seed})")

    # AGENT (PPO)
    agent_section = cfg.get("agent", {}) or {}
    if not isinstance(agent_section, dict):
        raise ValueError("La sección 'agent' debe ser un diccionario.")
    agent_kwargs = _strict_kwargs(PPOConfig, agent_section, "agent")
    if "seed" not in agent_kwargs:
        agent_kwargs["seed"] = int(base_seed)

    agent_cfg = PPOConfig(**agent_kwargs)
    agent = PPOAgent(agent_cfg, device=device)

    # TRAIN CFG
    train_section = cfg.get("train", {}) or {}
    if not isinstance(train_section, dict):
        raise ValueError("La sección 'train' debe ser un diccionario.")

    train_kwargs = _strict_kwargs(PPOTrainConfig, train_section, "train")
    if "seed" not in train_kwargs:
        train_kwargs["seed"] = int(base_seed)

    train_cfg = PPOTrainConfig(**train_kwargs)

    trainer = PPOTrainer(env, agent, cfg=train_cfg, device=device)

    # SNAPSHOT (Ley 4)
    snapshot_root = str(cfg.get("snapshot_dir", train_cfg.checkpoint_dir))
    snapshot_tag = f"{train_cfg.run_name}_seed{base_seed}_{timestamp()}"
    snapshot_path = os.path.join(snapshot_root, snapshot_tag)
    _ensure_dir(snapshot_path)

    extra_meta = {
        "device": str(device),
        "train_seed": int(base_seed),
        "env_seed": int(env_seed),
        "agent_seed": int(agent_cfg.seed),
        "stoch_enabled": bool(stoch_enabled),
        "stoch_meta": stoch_meta,
        "torch_flags": dict(torch_flags),
        "notes": "PPO run snapshot: resolved_config.yaml + run_meta.json",
    }
    _save_run_snapshot(snapshot_path, yaml_path=args.config, cfg=cfg, extra=extra_meta)

    # TRAIN
    result = trainer.train()
    print("Done:", result)


if __name__ == "__main__":
    main()