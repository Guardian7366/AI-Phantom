# scripts/evaluate_best_model.py
from __future__ import annotations

import argparse
import os
import json
from typing import Dict, Any, Optional, List

import torch
import yaml

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from controllers.evaluation_controller import EvaluationController

# Wrapper opcional (si existe)
try:
    from environments.maze.stochastic_wrapper import StochasticWrapper, StochasticConfig
except Exception:
    StochasticWrapper = None
    StochasticConfig = None


def _torch_load_compat(path: str, device: torch.device):
    # Compat: torch >=2.0 puede tener weights_only
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_path(path: str) -> str:
    p = (path or "").strip().strip('"').strip("'")
    p = os.path.expandvars(os.path.expanduser(p))
    return os.path.normpath(p)


def _list_subdirs_sorted_by_mtime(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    items: List[str] = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            items.append(full)
    items.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return items


def _is_run_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    # Heurística: contiene algún modelo típico
    candidates = ("best_model.pth", "last_model.pth", "best_model_lvl2.pth")
    return any(os.path.exists(os.path.join(path, f)) for f in candidates)


def _resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    """
    Acepta:
    - ruta directa al run folder (contiene best/last/best_lvl2)
    - ruta al root (ej: results/checkpoints) -> elige el subdir más reciente
    - ruta con placeholder "<run_id>" -> se ignora placeholder y toma el más reciente
    """
    p = _normalize_path(checkpoint_dir)

    if "<run_id>" in p or "%3Crun_id%3E" in p:
        root = p.replace("<run_id>", "").replace("%3Crun_id%3E", "").rstrip("\\/")
        if not root:
            root = os.path.normpath("results/checkpoints")
        runs = _list_subdirs_sorted_by_mtime(root)
        if not runs:
            raise FileNotFoundError(f"No hay runs dentro de: {root}")
        return runs[0]

    # Si ya es run dir válido
    if _is_run_dir(p):
        return p

    # Si es root con subdirs
    runs = _list_subdirs_sorted_by_mtime(p)
    if runs:
        return runs[0]

    # fallback: lo que sea que pasaron
    return p


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


def _apply_eval_torch_flags(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Mantiene compat con lo que hacemos en train.py:
    - deterministic (opt-in)
    - cudnn_benchmark (por defecto true si no deterministic)
    - allow_tf32 (por defecto true en cuda si no deterministic)
    """
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


def load_state_dict_safely(agent: DQNAgent, model_path: str) -> None:
    model_path = _normalize_path(model_path)
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


def _maybe_load_episode_seeds(path: Optional[str], episodes: int) -> Optional[List[int]]:
    if path is None:
        return None
    path = _normalize_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe episode_seeds_file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("episode_seeds_file debe ser un JSON list[int].")
    seeds = [int(x) for x in data]
    if len(seeds) < int(episodes):
        raise ValueError(f"episode_seeds_file tiene {len(seeds)} seeds pero episodes={episodes}.")
    return seeds[: int(episodes)]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint_dir", type=str, required=True,
                    help="Ruta al folder del run o al root (ej: results/checkpoints).")
    ap.add_argument("--which", type=str, default="best_lvl2",
                    choices=["best", "last", "best_lvl2"])

    ap.add_argument("--config", type=str, default="configs/maze_train.yaml",
                    help="YAML para construir env/agent consistente con train.")

    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--level", type=int, default=2, help="0/1/2 (recomendado 2).")
    ap.add_argument("--seed", type=int, default=123, help="Seed base para evaluación (si no hay episode_seeds).")

    ap.add_argument("--agent_seed", type=int, default=None,
                    help="Seed del agente (si no se pasa, usa agent.seed del YAML o root seed).")

    # Reproducibilidad EXACTA por episodio
    ap.add_argument("--episode_seeds_file", type=str, default=None,
                    help="JSON list[int] de seeds por episodio (sandbox 100% reproducible).")

    # Señal limpia en lvl2
    ap.add_argument("--freeze_pool", action="store_true",
                    help="Congela el pool lvl2 durante evaluación (recomendado).")
    ap.add_argument("--reset_pool_on_eval", action="store_true",
                    help="Resetea pool lvl2 UNA vez al inicio del batch (señal comparable).")

    # Sandbox / métricas extra
    ap.add_argument("--details", action="store_true",
                    help="Incluye episode_details (más pesado).")
    ap.add_argument("--track_grids", action="store_true",
                    help="Cuenta grids únicos evaluados (hash del grid).")
    ap.add_argument("--trajectories", action="store_true",
                    help="Guarda trayectorias (agent_pos por paso) dentro de episode_details.")

    # Output (Ley 4)
    ap.add_argument("--out", type=str, default=None,
                    help="Si se define, guarda JSON aquí. Si no, imprime por stdout.")

    # Robustez estocástica (opcional)
    ap.add_argument("--slip_prob", type=float, default=0.0,
                    help="Probabilidad de slip de acción (0.0 = determinista).")
    ap.add_argument("--slip_seed", type=int, default=999,
                    help="Seed del wrapper slip (independiente del env).")

    args = ap.parse_args()

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"No existe el directorio del run: {checkpoint_dir}")

    cfg = _load_yaml(args.config)
    device = _device_from_cfg(cfg)

    torch_flags = _apply_eval_torch_flags(cfg, device)

    env_section = cfg.get("env", {})
    agent_section = cfg.get("agent", {})
    if not isinstance(env_section, dict) or not isinstance(agent_section, dict):
        raise ValueError("Secciones env/agent deben ser diccionarios en el YAML.")

    env_cfg = MazeConfig(**_strict_kwargs(MazeConfig, env_section, "env"))
    agent_cfg = DQNConfig(**_strict_kwargs(DQNConfig, agent_section, "agent"))

    # Seed del agente: prioridad CLI > agent.seed YAML > root seed YAML > args.seed
    if args.agent_seed is not None:
        agent_cfg.seed = int(args.agent_seed)
    else:
        if hasattr(agent_cfg, "seed") and agent_cfg.seed is not None:
            # ya viene del YAML agent.seed
            pass
        elif "seed" in cfg:
            agent_cfg.seed = int(cfg["seed"])
        else:
            agent_cfg.seed = int(args.seed)

    # ⚠️ Recomendación: en lvl2 quieres freeze_pool=True para señal comparable
    if int(args.level) >= 2 and not bool(args.freeze_pool):
        print("[WARN] level>=2 y freeze_pool=False. Para métrica estable, usa --freeze_pool.")

    # Env base
    env_base = MazeEnvironment(env_cfg, rng_seed=int(args.seed))

    # Wrapper estocástico opcional
    env = env_base
    if float(args.slip_prob) > 0.0:
        if StochasticWrapper is None or StochasticConfig is None:
            raise ImportError("slip_prob>0 pero no se pudo importar StochasticWrapper/StochasticConfig.")
        wcfg = StochasticConfig(
            action_slip_prob=float(args.slip_prob),
            num_actions=int(agent_cfg.num_actions),
        )
        env = StochasticWrapper(env_base, cfg=wcfg, seed=int(args.slip_seed))

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

    ep_seeds = _maybe_load_episode_seeds(args.episode_seeds_file, int(args.episodes))

    stats = evaluator.evaluate(
        episodes=int(args.episodes),
        curriculum_level=int(args.level),
        seed=int(args.seed),
        freeze_pool=bool(args.freeze_pool),
        reset_pool_on_eval=bool(args.reset_pool_on_eval),
        episode_seeds=ep_seeds,
        return_episode_details=bool(args.details or args.trajectories),
        record_trajectories=bool(args.trajectories),
        track_grid_hashes=bool(args.track_grids),
    )

    out: Dict[str, Any] = {
        "checkpoint_dir": str(checkpoint_dir),
        "which": str(args.which),
        "model_path": str(_normalize_path(model_path)),
        "device": str(agent.device),
        "episodes": int(args.episodes),
        "level": int(args.level),
        "seed": int(args.seed),
        "episode_seeds_file": None if args.episode_seeds_file is None else str(_normalize_path(args.episode_seeds_file)),
        "agent_seed": int(agent_cfg.seed),
        "freeze_pool": bool(args.freeze_pool),
        "reset_pool_on_eval": bool(args.reset_pool_on_eval),
        "slip_prob": float(args.slip_prob),
        "slip_seed": int(args.slip_seed),
        "torch_flags": dict(torch_flags),
        **stats,
    }

    if args.out:
        out_path = _normalize_path(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()