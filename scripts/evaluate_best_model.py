import yaml
import json
from pathlib import Path

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import ReplayBuffer
from controllers.evaluation_controller import EvaluationController


# -------------------------------------------------
# Utils
# -------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print("\n=== Evaluación cuantitativa: Best Model ===\n")

    config_path = "configs/maze_evaluation.yaml"
    cfg = load_config(config_path)

    # ----------------------------
    # Environment factory
    # ----------------------------

    def env_factory():
        return MazeEnvironment(cfg)

    # ----------------------------
    # Agent factory
    # ----------------------------

    def agent_factory():
        replay_buffer = ReplayBuffer(capacity=1)  # dummy buffer

        agent = DQNAgent(
            state_dim=6,
            action_dim=4,
            replay_buffer=replay_buffer,
            gamma=cfg["agent"]["gamma"],
            lr=cfg["agent"]["learning_rate"],
            batch_size=cfg["agent"]["batch_size"],
            target_update_freq=cfg["agent"]["target_update_frequency"],
        )

        agent.set_mode(training=False)
        return agent

    # ----------------------------
    # Evaluation
    # ----------------------------

    evaluator = EvaluationController(
        env_factory=env_factory,
        agent_factory=agent_factory,
        config=cfg,
    )

    model_path = cfg["model"]["path"]

    results = evaluator.evaluate_checkpoint(model_path)

    # ----------------------------
    # Output
    # ----------------------------

    print("=== RESULTADOS ===")
    for k, v in results.items():
        print(f"{k:20s}: {v}")

    output_path = Path("results") / "best_model" / "evaluation_summary.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluación guardada en: {output_path}\n")


if __name__ == "__main__":
    main()
