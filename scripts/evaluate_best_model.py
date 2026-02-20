# scripts/evaluate_best_model.py
import argparse
import os
import json
from typing import Dict, Any

import torch
import yaml

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from controllers.evaluation_controller import EvaluationController


def _torch_load_compat(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_checkpoint_dir(path: str) -> str:
    if path is None:
        return ""
    p = path.strip().strip('"').strip("'")
    p = os.path.expandvars(os.path.expanduser(p))
    return os.path.normpath(p)


def _list_run_dirs(checkpoints_root: str):
    if not os.path.isdir(checkpoints_root):
        return []
    items = []
    for name in os.listdir(checkpoints_root):
        full = os.path.join(checkpoints_root, name)
        if os.path.isdir(full):
            items.append(full)
    items.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return items


def _resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    checkpoint_dir = _normalize_checkpoint_dir(checkpoint_dir)

    if os.path.basename(checkpoint_dir).lower() == "checkpoints" and os.path.isdir(checkpoint_dir):
        runs = _list_run_dirs(checkpoint_dir)
        if not runs:
            raise FileNotFoundError(f"No hay runs dentro de: {checkpoint_dir}")
        return runs[0]

    if "<run_id>" in checkpoint_dir or "%3Crun_id%3E" in checkpoint_dir:
        root = checkpoint_dir.replace("<run_id>", "").replace("%3Crun_id%3E", "")
        root = root.rstrip("\\/")
        if not root:
            root = os.path.normpath("results/checkpoints")
        runs = _list_run_dirs(root)
        if not runs:
            raise FileNotFoundError(f"No hay runs dentro de: {root}")
        return runs[0]

    return checkpoint_dir


def _load_yaml(path: str) -> Dict[str, Any]:
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


def load_state_dict_safely(agent: DQNAgent, model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    sd = _torch_load_compat(model_path, agent.device)
    if not isinstance(sd, dict):
        raise ValueError(f"El archivo no parece state_dict (dict): {type(sd)} en {model_path}")

    agent.q.load_state_dict(sd)
    agent.q_tgt.load_state_dict(sd)
    agent.q.eval()
    agent.q_tgt.eval()


def _pick_model_file(which: str) -> str:
    if which == "best":
        return "best_model.pth"
    if which == "last":
        return "last_model.pth"
    return "best_model_lvl2.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="Ruta al folder del run o a results/checkpoints")
    ap.add_argument("--which", type=str, default="best_lvl2",
                    choices=["best", "last", "best_lvl2"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--level", type=int, default=2,
                    help="0/1/2 (recomendado 2).")
    ap.add_argument("--seed", type=int, default=123,
                    help="Seed base para evaluación (episodios usan seeds derivados).")
    ap.add_argument("--agent_seed", type=int, default=None,
                    help="Seed del agente (si no se pasa, usa cfg['seed'] si existe).")
    ap.add_argument("--config", type=str, default="configs/maze_train.yaml",
                    help="YAML para construir env/agent consistente con train.")
    ap.add_argument("--freeze_pool", action="store_true",
                    help="Congela el pool lvl2 durante evaluación (recomendado).")
    args = ap.parse_args()

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"No existe el directorio del run: {checkpoint_dir}")

    cfg = _load_yaml(args.config)

    device_str = str(cfg.get("device", "auto")).lower()
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str in ("cuda", "cpu"):
        device = torch.device(device_str)
    else:
        raise ValueError("device debe ser: auto/cuda/cpu")

    env_section = cfg.get("env", {})
    agent_section = cfg.get("agent", {})
    if not isinstance(env_section, dict) or not isinstance(agent_section, dict):
        raise ValueError("Secciones env/agent deben ser diccionarios.")

    env_cfg = MazeConfig(**_strict_kwargs(MazeConfig, env_section, "env"))
    agent_cfg = DQNConfig(**_strict_kwargs(DQNConfig, agent_section, "agent"))

    # Seed del agente
    if args.agent_seed is not None:
        agent_cfg.seed = int(args.agent_seed)
    elif "seed" in cfg:
        agent_cfg.seed = int(cfg["seed"])

    env = MazeEnvironment(env_cfg, rng_seed=int(args.seed))
    agent = DQNAgent(agent_cfg, device=device)

    model_file = _pick_model_file(args.which)
    model_path = os.path.join(checkpoint_dir, model_file)

    if args.which == "best_lvl2" and not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No existe el modelo: {model_path}\n"
            f"Revisa que en {checkpoint_dir} existan: last_model.pth / best_model.pth / best_model_lvl2.pth"
        )

    load_state_dict_safely(agent, model_path)

    evaluator = EvaluationController(env, agent)
    stats = evaluator.evaluate(
        episodes=int(args.episodes),
        curriculum_level=int(args.level),
        seed=int(args.seed),
        freeze_pool=bool(args.freeze_pool),  # por flag
    )

    print(json.dumps({
        "checkpoint_dir": checkpoint_dir,
        "which": args.which,
        "model_path": model_path,
        "device": str(agent.device),
        "episodes": int(args.episodes),
        "level": int(args.level),
        "seed": int(args.seed),
        "agent_seed": None if args.agent_seed is None else int(args.agent_seed),
        "freeze_pool": bool(args.freeze_pool),
        **stats
    }, indent=2))


if __name__ == "__main__":
    main()