from __future__ import annotations
from typing import Iterable
from ai_phantom.envs.maze.maze_env import MazeEnv

def sync_horizon(envs: Iterable[MazeEnv], rollout_len: int, *, name: str = "") -> None:
    rl = int(rollout_len)
    for i, env in enumerate(envs):
        if int(env.cfg.max_steps) != rl:
            tag = f"{name} " if name else ""
            print(f"⚠️ {tag}Horizon mismatch env[{i}]: max_steps={env.cfg.max_steps} vs rollout_len={rl}. Forzando max_steps=rollout_len.")
            env.cfg.max_steps = rl