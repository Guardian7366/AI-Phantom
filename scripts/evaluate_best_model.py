# scripts/evaluate_best_model.py
# Evaluación cuantitativa del modelo global seleccionado

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

    # -------------------------------------------------
    # Ruta oficial del best model
    # -------------------------------------------------

    model_path = Path("results") / "best_model" / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró best_model.pth en {model_path}"
        )

    # -------------------------------------------------
    # Factories
    # -------------------------------------------------

    # -------------------------------------------------
    # Factories (CORRECTAS)
    # -------------------------------------------------

    env_cfg = cfg["environment"]

    def env_factory():
        return MazeEnvironment(env_cfg)

    def agent_factory():
        temp_env = MazeEnvironment(env_cfg)

        replay_buffer = ReplayBuffer(capacity=1)

        agent = DQNAgent(
            state_dim=temp_env.state_dim,
            action_dim=temp_env.action_space_n,
            replay_buffer=replay_buffer,
        )

        agent.set_mode(training=False)
        return agent


    # -------------------------------------------------
    # Evaluación
    # -------------------------------------------------

    evaluator = EvaluationController(
        env_factory=env_factory,
        agent_factory=agent_factory,
        config=cfg,
    )

    results = evaluator.evaluate_checkpoint(str(model_path))

    # -------------------------------------------------
    # Output en consola
    # -------------------------------------------------

    print("=== RESULTADOS ===\n")
    for k, v in results.items():
        print(f"{k:20s}: {v}")

    # -------------------------------------------------
    # Guardar resultados
    # -------------------------------------------------

    output_path = Path("results") / "best_model" / "evaluation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluación guardada en: {output_path}\n")


if __name__ == "__main__":
    main()
