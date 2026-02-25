from __future__ import annotations

from typing import Iterable, Literal
from ai_phantom.envs.maze.maze_env import MazeEnv


HorizonMode = Literal["warn", "force", "assert"]


def sync_horizon(
    envs: Iterable[MazeEnv],
    rollout_len: int,
    *,
    name: str = "",
    mode: HorizonMode = "warn",
) -> None:
    """
    Chequea consistencia entre rollout_len (buffer) y max_steps (episodio).

    - warn  : solo imprime warning (SAFE default)
    - force : setea max_steps = rollout_len
    - assert: lanza error si hay mismatch

    Nota: no es requisito que max_steps == rollout_len para PPO.
    """
    rl = int(rollout_len)
    for i, env in enumerate(envs):
        ms = int(env.cfg.max_steps)
        if ms != rl:
            tag = f"{name} " if name else ""
            msg = (
                f"⚠️ {tag}Horizon mismatch env[{i}]: max_steps={ms} vs rollout_len={rl}."
            )

            if mode == "warn":
                print(msg + " (no se fuerza; mode='warn')")
            elif mode == "force":
                print(msg + " Forzando max_steps=rollout_len (mode='force').")
                env.cfg.max_steps = rl
            elif mode == "assert":
                raise RuntimeError(msg + " (mode='assert')")
            else:
                print(msg + f" (mode desconocido={mode!r}; no se fuerza)")