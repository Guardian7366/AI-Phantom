import argparse
import os
import json
import torch

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="results/inference/inference_episode.json")
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

    obs, info = env.reset(curriculum_level=args.level, seed=args.seed)
    done = False
    trunc = False

    episode = {
        "grid": env.grid.tolist(),
        "start": list(env.agent_pos),
        "goal": list(env.goal_pos),
        "steps": []
    }

    while not (done or trunc):
        a = agent.act(obs, deterministic=True)
        prev = env.agent_pos
        obs, r, done, trunc, info = env.step(a)
        episode["steps"].append({
            "action": int(a),
            "reward": float(r),
            "from": list(prev),
            "to": list(env.agent_pos),
            "done": bool(done),
            "trunc": bool(trunc),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(episode, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
