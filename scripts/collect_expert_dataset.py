# scripts/collect_expert_dataset.py
from __future__ import annotations

import os
import json
import argparse
from dataclasses import asdict
from typing import Dict, Any, List, Optional

import numpy as np
import yaml

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from environments.maze.maze_generator import MazeExpertPlanner, ExpertConfig, rollout_expert_episode


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _build_env_from_config(cfg: Dict[str, Any]) -> MazeEnvironment:
    env_cfg = cfg.get("env", {}) or {}
    # MazeConfig acepta families: Optional[Dict[str,Any]]
    mc = MazeConfig(**env_cfg)
    env_seed = int(cfg.get("env_seed", cfg.get("seed", 0)))
    return MazeEnvironment(config=mc, rng_seed=env_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/maze_train.yaml")
    ap.add_argument("--out_dir", type=str, default="results/expert_dataset")
    ap.add_argument("--out_name", type=str, default="maze_expert_bc")
    ap.add_argument("--episodes", type=int, default=4000)

    # mezcla por niveles (para BC conviene meter lvl2 fuerte)
    ap.add_argument("--p_lvl0", type=float, default=0.10)
    ap.add_argument("--p_lvl1", type=float, default=0.25)
    ap.add_argument("--p_lvl2", type=float, default=0.65)

    ap.add_argument("--freeze_pool", action="store_true", default=False)
    ap.add_argument("--seed0", type=int, default=12345)

    # experto
    ap.add_argument("--algo", type=str, default="bfs", choices=["bfs", "astar"])
    ap.add_argument("--max_nodes", type=int, default=200000)

    # dataset
    ap.add_argument("--max_steps_cap", type=int, default=256)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    env = _build_env_from_config(cfg)

    planner = MazeExpertPlanner(ExpertConfig(algorithm=str(args.algo), max_nodes=int(args.max_nodes)))

    _ensure_dir(args.out_dir)
    out_npz = os.path.join(args.out_dir, f"{args.out_name}.npz")
    out_meta = os.path.join(args.out_dir, f"{args.out_name}_meta.json")

    # distribución de niveles
    probs = np.array([args.p_lvl0, args.p_lvl1, args.p_lvl2], dtype=np.float64)
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0:
        probs = np.array([0.10, 0.25, 0.65], dtype=np.float64)
        s = float(probs.sum())
    probs /= s

    rng = np.random.default_rng(int(args.seed0))

    obs_buf: List[np.ndarray] = []
    act_buf: List[np.int64] = []

    stats = {
        "episodes_requested": int(args.episodes),
        "episodes_collected": 0,
        "episodes_failed_no_path": 0,
        "episodes_failed_other": 0,
        "steps_total": 0,
        "steps_mean": 0.0,
        "reached_goal_rate": 0.0,
        "by_level": {
            "0": {"eps": 0, "ok": 0, "steps": 0},
            "1": {"eps": 0, "ok": 0, "steps": 0},
            "2": {"eps": 0, "ok": 0, "steps": 0},
        },
        "env_cfg": cfg.get("env", {}),
        "expert_cfg": {"algorithm": str(args.algo), "max_nodes": int(args.max_nodes)},
    }

    ok_eps = 0
    total_steps = 0

    for ep in range(int(args.episodes)):
        lvl = int(rng.choice([0, 1, 2], p=probs))
        seed = int(args.seed0 + ep)  # determinista por ep

        stats["by_level"][str(lvl)]["eps"] += 1

        try:
            roll = rollout_expert_episode(
                env,
                planner,
                curriculum_level=lvl,
                seed=seed,
                freeze_pool=bool(args.freeze_pool),
                max_steps_cap=int(args.max_steps_cap),
            )
        except Exception:
            stats["episodes_failed_other"] += 1
            continue

        if not roll.get("obs_list"):
            stats["episodes_failed_no_path"] += 1
            continue

        # Para BC: solo guardamos (obs, action) por paso
        obs_list = roll["obs_list"]
        act_list = roll["act_list"]

        # defensivo: si por algún bug path_to_actions devuelve []
        if len(act_list) == 0 or len(obs_list) != len(act_list):
            stats["episodes_failed_other"] += 1
            continue

        # Guardado: obs float32 (3,H,W) y actions int64
        obs_buf.extend([np.asarray(o, dtype=np.float32) for o in obs_list])
        act_buf.extend([np.int64(a) for a in act_list])

        stats["episodes_collected"] += 1
        steps = int(len(act_list))
        total_steps += steps

        if bool(roll.get("reached_goal", False)):
            ok_eps += 1
            stats["by_level"][str(lvl)]["ok"] += 1
        stats["by_level"][str(lvl)]["steps"] += steps

    stats["steps_total"] = int(total_steps)
    stats["steps_mean"] = float(total_steps / max(1, stats["episodes_collected"]))
    stats["reached_goal_rate"] = float(ok_eps / max(1, stats["episodes_collected"]))

    # Serializar dataset
    obs_arr = np.stack(obs_buf, axis=0) if len(obs_buf) else np.zeros((0, 3, env.size, env.size), dtype=np.float32)
    act_arr = np.asarray(act_buf, dtype=np.int64) if len(act_buf) else np.zeros((0,), dtype=np.int64)

    np.savez_compressed(out_npz, obs=obs_arr, actions=act_arr)

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved dataset: {out_npz}")
    print(f"[OK] Saved meta:    {out_meta}")
    print(f"[OK] Samples: obs={obs_arr.shape} actions={act_arr.shape}")


if __name__ == "__main__":
    main()