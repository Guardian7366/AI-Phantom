import argparse
import os
import torch

from environments.maze.maze_env import MazeEnvironment, MazeConfig
from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from controllers.evaluation_controller import EvaluationController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", type=str, required=True)
    ap.add_argument("--which", type=str, default="best", choices=["best", "last"])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=999)
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
    agent.q_tgt.eval()

    evaluator = EvaluationController(env, agent)
    stats = evaluator.evaluate(episodes=args.episodes, curriculum_level=args.level, seed=args.seed)

    print("SMOKE TEST OK")
    print(stats)


if __name__ == "__main__":
    main()
