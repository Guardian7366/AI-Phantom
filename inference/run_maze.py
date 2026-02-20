from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Tuple

import numpy as np
import yaml

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNConfig
from inference.model_loader import load_dqn_agent


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
    unknown = set(section.keys()) - allowed
    if unknown:
        raise KeyError(
            f"Claves desconocidas en '{section_name}': {sorted(list(unknown))}. "
            f"Permitidas: {sorted(list(allowed))}"
        )
    return dict(section)


def run_one_episode(env: MazeEnvironment, agent, curriculum_level: int, seed: int) -> Dict[str, Any]:
    obs, info = env.reset(curriculum_level=curriculum_level, seed=seed)
    done = False
    trunc = False

    episode = {
        "grid": env.grid.tolist(),
        "start": list(env.agent_pos),
        "goal": list(env.goal_pos),
        "steps": [],
        "meta": {
            "curriculum_level": int(curriculum_level),
            "seed": int(seed),
        }
    }

    while not (done or trunc):
        a = agent.act(obs, deterministic=True)
        prev = env.agent_pos
        obs, r, done, trunc, info = env.step(a)
        episode["steps"].append({
            "action": int(a),
            "reward": float(r),
            "from": list(prev),
            "to": list(env.agent_pos),
            "done": bool(done),
            "trunc": bool(trunc),
        })

    episode["meta"]["success"] = bool(done)
    episode["meta"]["num_steps"] = len(episode["steps"])
    return episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/maze_inference.yaml",
                    help="Ruta a configs/maze_inference.yaml")
    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="Ruta al folder del run: results/checkpoints/<run_id>/")
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    seed = int(cfg.get("seed", 123))
    device_str = str(cfg.get("device", "auto"))

    # ENV (estricto)
    env_section = cfg.get("env", {})
    if not isinstance(env_section, dict):
        raise ValueError("La sección 'env' debe ser un diccionario.")
    env_cfg = MazeConfig(**_strict_kwargs(MazeConfig, env_section, "env"))
    env = MazeEnvironment(env_cfg, rng_seed=seed)

    # AGENT config (opcional en inference yaml; si no existe, usa DQNConfig())
    agent_section = cfg.get("agent", {})
    if agent_section is None:
        agent_section = {}
    if not isinstance(agent_section, dict):
        raise ValueError("La sección 'agent' debe ser un diccionario si existe.")
    agent_cfg = DQNConfig(**_strict_kwargs(DQNConfig, agent_section, "agent")) if agent_section else DQNConfig()

    # INFERENCE params
    inf_section = cfg.get("inference", {})
    if not isinstance(inf_section, dict):
        raise ValueError("La sección 'inference' debe ser un diccionario.")
    allowed_inf = {"curriculum_level", "output_json"}
    unknown_inf = set(inf_section.keys()) - allowed_inf
    if unknown_inf:
        raise KeyError(f"Claves desconocidas en 'inference': {sorted(list(unknown_inf))}")

    curriculum_level = int(inf_section.get("curriculum_level", 2))
    out_path = str(inf_section.get("output_json", "results/inference/inference_episode.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    agent, meta = load_dqn_agent(
        checkpoint_dir=args.checkpoint_dir,
        which=args.which,
        device_str=device_str,
        agent_cfg=agent_cfg,
    )

    episode = run_one_episode(env, agent, curriculum_level=curriculum_level, seed=seed)
    episode["meta"]["model_path"] = meta.model_path
    episode["meta"]["device"] = str(meta.device)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")
    print(json.dumps(episode["meta"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
