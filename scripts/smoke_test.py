from __future__ import annotations

from ai_phantom.core import set_global_seed
from ai_phantom.envs.maze import MazeConfig, MazeEnv
from ai_phantom.planners.bfs import bfs_plan, path_to_actions


def run_once(phase: int, seed: int) -> None:
    cfg = MazeConfig(
        height=12,
        width=12,
        use_walls=False,      # hoy: simple y estable
        max_steps=256,
        min_manhattan=6,
    )
    env = MazeEnv(cfg, seed=seed)
    obs, info = env.reset(seed=seed, phase=phase)

    print("\n=== RESET ===")
    print(f"phase={phase} seed={seed}")
    print("agent:", info["agent"], "goal:", info["goal"])
    print(env.render())

    path = bfs_plan(env.walls, env.agent, env.goal)
    if path is None:
        raise RuntimeError("BFS no encontró ruta (no debería pasar si no hay paredes).")

    actions = path_to_actions(path)

    total_reward = 0.0
    done = False
    for a in actions:
        obs, r, done, info = env.step(a)
        total_reward += r
        if done:
            break

    print("\n=== END ===")
    print("done:", done, "reached:", info["reached"], "t:", info["t"])
    print("total_reward:", round(total_reward, 4))
    print(env.render())

    if not info["reached"]:
        raise RuntimeError("Smoke test falló: no llegó a la meta.")


def main() -> None:
    set_global_seed(123)

    # Fase 0
    run_once(phase=0, seed=42)

    # Fase 1 (varios seeds)
    for s in [1, 2, 3, 4, 5]:
        run_once(phase=1, seed=s)

    print("\nOK ✅ Smoke test pasó.")


if __name__ == "__main__":
    main()