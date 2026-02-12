# scripts/evaluate_checkpoints.py
# Evaluación cuantitativa de todos los checkpoints DQN
# Arquitectura limpia basada en factories

import os
import json
import yaml
import torch
import argparse

from controllers.evaluation_controller import EvaluationController
from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import ReplayBuffer


# -------------------------------------------------
# ARGPARSE
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/maze_evaluation.yaml",
        help="Ruta al archivo YAML de evaluación"
    )
    return parser.parse_args()


# -------------------------------------------------
# VALIDACIÓN DE CHECKPOINT
# -------------------------------------------------

def validate_checkpoint_compatibility(agent: DQNAgent, checkpoint_path: str) -> bool:
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    weight_keys = [k for k in state_dict.keys() if k.endswith("weight")]
    if not weight_keys:
        return False

    first_w = state_dict[weight_keys[0]]
    last_w = state_dict[weight_keys[-1]]

    state_dim_ok = first_w.shape[1] == agent.state_dim
    action_dim_ok = last_w.shape[0] == agent.action_dim

    return state_dim_ok and action_dim_ok


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    paths_cfg = config["paths"]

    checkpoint_dir = paths_cfg["checkpoint_dir"]
    results_path = paths_cfg["results_path"]

    checkpoint_files = sorted(
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")
    )

    if not checkpoint_files:
        raise RuntimeError("No se encontraron checkpoints .pth")

    all_results = []

    for ckpt in checkpoint_files:

        # -------------------------------------------------
        # Factories (DISEÑO CORRECTO)
        # -------------------------------------------------

        def env_factory():
            return MazeEnvironment(env_cfg)

        def agent_factory():
            temp_env = MazeEnvironment(env_cfg)
            replay_buffer = ReplayBuffer(capacity=1)

            return DQNAgent(
                state_dim=temp_env.state_dim,
                action_dim=temp_env.action_space_n,
                replay_buffer=replay_buffer,
            )

        # Instancia temporal SOLO para validar shapes
        temp_env = MazeEnvironment(env_cfg)
        temp_agent = DQNAgent(
            state_dim=temp_env.state_dim,
            action_dim=temp_env.action_space_n,
            replay_buffer=ReplayBuffer(capacity=1),
        )

        ckpt_path = os.path.join(checkpoint_dir, ckpt)

        if not validate_checkpoint_compatibility(temp_agent, ckpt_path):
            print(f"[SKIP] Checkpoint incompatible: {ckpt}")
            continue

        # -------------------------------------------------
        # Evaluación limpia basada en factories
        # -------------------------------------------------

        evaluator = EvaluationController(
            env_factory=env_factory,
            agent_factory=agent_factory,
            config=config,
        )

        summary = evaluator.evaluate_checkpoint(ckpt_path)
        summary["model"] = ckpt

        all_results.append(summary)

    # -------------------------------------------------
    # TABLA EN CONSOLA
    # -------------------------------------------------

    print("\n=== COMPARATIVA DE MODELOS ===\n")

    header = (
        f"{'Model':25} | {'Success':8} | {'MeanLen':8} | "
        f"{'MeanRew':8} | {'LenP75':8}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['model']:25} | "
            f"{r['success_rate']:.3f}   | "
            f"{r['mean_length']:.2f}   | "
            f"{r['mean_reward']:.2f}   | "
            f"{r['length_p75']:.2f}"
        )

    # -------------------------------------------------
    # Guardar JSON
    # -------------------------------------------------

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados guardados en {results_path}\n")


if __name__ == "__main__":
    main()
