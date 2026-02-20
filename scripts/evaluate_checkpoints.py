import argparse
import os
import json
import torch

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from controllers.evaluation_controller import EvaluationController


def eval_one_run(run_dir: str, which: str, episodes: int, level: int, seed: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = MazeEnvironment(MazeConfig(), rng_seed=seed)
    agent = DQNAgent(DQNConfig(), device=device)

    model_file = "best_model.pth" if which == "best" else "last_model.pth"
    model_path = os.path.join(run_dir, model_file)
    if not os.path.exists(model_path):
        return None

    sd = torch.load(model_path, map_location=device)
    agent.q.load_state_dict(sd)
    agent.q_tgt.load_state_dict(sd)
    agent.q.eval()
    agent.q_tgt.eval()

    evaluator = EvaluationController(env, agent)
    stats = evaluator.evaluate(episodes=episodes, curriculum_level=level, seed=seed)
    return {"run_dir": run_dir, "model": which, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints_root", type=str, default="results/checkpoints")
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="results/runs/checkpoints_eval_summary.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    runs = []
    if not os.path.isdir(args.checkpoints_root):
        raise FileNotFoundError(f"No existe checkpoints_root: {args.checkpoints_root}")

    for name in sorted(os.listdir(args.checkpoints_root)):
        run_dir = os.path.join(args.checkpoints_root, name)
        if not os.path.isdir(run_dir):
            continue
        res = eval_one_run(run_dir, args.which, args.episodes, args.level, args.seed)
        if res is not None:
            runs.append(res)

    payload = {
        "checkpoints_root": args.checkpoints_root,
        "which": args.which,
        "episodes": args.episodes,
        "level": args.level,
        "seed": args.seed,
        "runs": runs
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved: {args.out}")
    print(f"Runs evaluated: {len(runs)}")


if __name__ == "__main__":
    main()
