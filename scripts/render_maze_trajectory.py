import argparse
import os
import time
import torch

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig


def render_ascii(env: MazeEnvironment) -> str:
    lines = []
    for r in range(env.size):
        row = []
        for c in range(env.size):
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
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--delay", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MazeEnvironment(MazeConfig(), rng_seed=args.seed)
    agent = DQNAgent(DQNConfig(), device=device)

    model_file = "best_model.pth" if args.which == "best" else "last_model.pth"
    model_path = os.path.join(args.checkpoint_dir, model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    sd = torch.load(model_path, map_location=device)
    agent.q.load_state_dict(sd)
    agent.q_tgt.load_state_dict(sd)
    agent.q.eval()

    obs, info = env.reset(curriculum_level=args.level, seed=args.seed)
    done = False
    trunc = False
    step = 0

    print("Initial:")
    print(render_ascii(env))
    print("-" * 20)

    while not (done or trunc):
        a = agent.act(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(a)
        step += 1
        print(f"Step {step} | a={a} r={r:.3f} done={done} trunc={trunc}")
        print(render_ascii(env))
        print("-" * 20)
        time.sleep(args.delay)

    print("Finished.")


if __name__ == "__main__":
    main()
