import argparse
from environments.maze.maze_env import MazeEnvironment, MazeConfig


def render_ascii(env: MazeEnvironment) -> str:
    H = env.size
    W = env.size
    lines = []
    for r in range(H):
        row = []
        for c in range(W):
            if (r, c) == env.agent_pos:
                row.append("A")
            elif (r, c) == env.goal_pos:
                row.append("G")
            elif env.grid[r, c] == 1:
                row.append("#")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = MazeEnvironment(MazeConfig(), rng_seed=args.seed)
    env.reset(curriculum_level=args.level, seed=args.seed)

    print(render_ascii(env))


if __name__ == "__main__":
    main()
