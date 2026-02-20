import argparse
import os
import json
import torch
import numpy as np

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="results/trajectories/trajectories.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

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

    rng = np.random.default_rng(args.seed)

    trajectories = []
    for ep in range(args.episodes):
        obs, info = env.reset(curriculum_level=args.level, seed=int(rng.integers(0, 10_000_000)))
        done = False
        trunc = False

        traj = {
            "episode": ep,
            "grid": env.grid.tolist(),
            "start": list(env.agent_pos),
            "goal": list(env.goal_pos),
            "steps": []
        }

        while not (done or trunc):
            a = agent.act(obs, deterministic=True)
            prev_pos = env.agent_pos
            obs, r, done, trunc, info = env.step(a)
            traj["steps"].append({
                "action": int(a),
                "reward": float(r),
                "from": list(prev_pos),
                "to": list(env.agent_pos),
                "done": bool(done),
                "trunc": bool(trunc),
            })

        trajectories.append(traj)

    payload = {
        "checkpoint_dir": args.checkpoint_dir,
        "which": args.which,
        "level": args.level,
        "episodes": args.episodes,
        "seed": args.seed,
        "trajectories": trajectories
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
