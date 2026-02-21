# scripts/run_inference.py
from __future__ import annotations

import argparse
import os
import json
from typing import Dict, Any, Optional

import torch
import yaml

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig

try:
    from environments.maze.stochastic_wrapper import StochasticWrapper, StochasticConfig
except Exception:
    StochasticWrapper = None
    StochasticConfig = None


def _normalize_path(path: str) -> str:
    p = (path or "").strip().strip('"').strip("'")
    p = os.path.expandvars(os.path.expanduser(p))
    return os.path.normpath(p)


def _torch_load_compat(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_yaml(path: str) -> Dict[str, Any]:
    path = _normalize_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el YAML: {path}")
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


def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    device_str = str(cfg.get("device", "auto")).lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str in ("cuda", "cpu"):
        return torch.device(device_str)
    raise ValueError("device debe ser: auto/cuda/cpu")


def _pick_model_file(which: str) -> str:
    if which == "best":
        return "best_model.pth"
    if which == "last":
        return "last_model.pth"
    return "best_model_lvl2.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--which", type=str, default="best_lvl2", choices=["best", "last", "best_lvl2"])

    ap.add_argument("--config", type=str, default="configs/maze_train.yaml",
                    help="YAML para construir env/agent consistente.")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--episode_seed", type=int, default=123,
                    help="Seed por episodio para reset(env). (Sandbox reproducible)")
    ap.add_argument("--agent_seed", type=int, default=None,
                    help="Seed del agente. Si no, usa agent.seed del YAML o root seed.")
    ap.add_argument("--out", type=str, default="results/inference/inference_episode.json")

    # Slip opcional
    ap.add_argument("--slip_prob", type=float, default=0.0)
    ap.add_argument("--slip_seed", type=int, default=999)

    # Hard cap por seguridad (si quieres)
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Override del máximo de pasos del episodio. Default: env.cfg.max_steps")

    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    device = _device_from_cfg(cfg)

    env_section = cfg.get("env", {})
    agent_section = cfg.get("agent", {})
    if not isinstance(env_section, dict) or not isinstance(agent_section, dict):
        raise ValueError("Secciones env/agent deben ser diccionarios en el YAML.")

    env_cfg = MazeConfig(**_strict_kwargs(MazeConfig, env_section, "env"))
    agent_cfg = DQNConfig(**_strict_kwargs(DQNConfig, agent_section, "agent"))

    # Agent seed: CLI > agent.seed YAML > root seed > episode_seed
    if args.agent_seed is not None:
        agent_cfg.seed = int(args.agent_seed)
    else:
        if hasattr(agent_cfg, "seed") and agent_cfg.seed is not None:
            pass
        elif "seed" in cfg:
            agent_cfg.seed = int(cfg["seed"])
        else:
            agent_cfg.seed = int(args.episode_seed)

    env_base = MazeEnvironment(env_cfg, rng_seed=int(args.episode_seed))

    env = env_base
    if float(args.slip_prob) > 0.0:
        if StochasticWrapper is None or StochasticConfig is None:
            raise ImportError("slip_prob>0 pero no se pudo importar StochasticWrapper/StochasticConfig.")
        wcfg = StochasticConfig(action_slip_prob=float(args.slip_prob), num_actions=int(agent_cfg.num_actions))
        env = StochasticWrapper(env_base, cfg=wcfg, seed=int(args.slip_seed))

    agent = DQNAgent(agent_cfg, device=device)

    model_file = _pick_model_file(args.which)
    model_path = os.path.join(_normalize_path(args.checkpoint_dir), model_file)
    model_path = _normalize_path(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    sd = _torch_load_compat(model_path, device)
    if not isinstance(sd, dict):
        raise ValueError(f"El archivo no parece state_dict (dict): {type(sd)} en {model_path}")

    agent.q.load_state_dict(sd)
    agent.q_tgt.load_state_dict(sd)
    agent.q.eval()
    agent.q_tgt.eval()

    obs, info = env.reset(curriculum_level=int(args.level), seed=int(args.episode_seed))

    done = False
    trunc = False

    max_steps = int(args.max_steps) if args.max_steps is not None else int(getattr(getattr(env, "cfg", None), "max_steps", 0) or 0)
    if max_steps <= 0:
        max_steps = 10_000

    episode: Dict[str, Any] = {
        "checkpoint_dir": _normalize_path(args.checkpoint_dir),
        "which": str(args.which),
        "model_path": str(model_path),
        "device": str(device),
        "level": int(args.level),
        "episode_seed": int(args.episode_seed),
        "agent_seed": int(agent_cfg.seed),
        "slip_prob": float(args.slip_prob),
        "slip_seed": int(args.slip_seed),
        "max_steps": int(max_steps),
        "grid": getattr(env, "grid", None).tolist() if getattr(env, "grid", None) is not None else None,
        "start": [int(x) for x in getattr(env, "agent_pos", (0, 0))],
        "goal": [int(x) for x in getattr(env, "goal_pos", (0, 0))],
        "steps": [],
    }

    steps = 0
    while not (done or trunc):
        a = agent.act(obs, deterministic=True)
        prev = getattr(env, "agent_pos", (0, 0))

        obs, r, done, trunc, info2 = env.step(int(a))
        steps += 1

        episode["steps"].append({
            "t": int(steps),
            "action": int(a),
            "reward": float(r),
            "from": [int(prev[0]), int(prev[1])],
            "to": [int(getattr(env, "agent_pos", (0, 0))[0]), int(getattr(env, "agent_pos", (0, 0))[1])],
            "done": bool(done),
            "truncated": bool(trunc),
            "info": dict(info2) if isinstance(info2, dict) else {},
        })

        if steps >= max_steps and not done and not trunc:
            trunc = True
            break

    out_path = _normalize_path(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()