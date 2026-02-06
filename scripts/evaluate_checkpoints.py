# scripts/evaluate_checkpoints.py
# Script de evaluación cuantitativa de múltiples checkpoints DQN
# Decisiones clave:
# - Configuración externa vía YAML (sin hardcode)
# - Factory simple de entorno basada en config
# - Validación básica de compatibilidad por shapes de la red
# - Inferencia pura: sin replay buffer funcional, sin exploración

import os
import json
import yaml
import numpy as np
import torch

from controllers.evaluation_controller import EvaluationController
from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import ReplayBuffer


# --------- CONFIG ---------

CONFIG_PATH = "configs/maze_inference.yaml"


# --------- FACTORY ---------

def make_maze_env(config: dict):
    """
    Factory mínima de entorno.
    Asume que MazeEnvironment acepta kwargs definidos en el YAML.
    """
    return MazeEnvironment(**config)


# --------- VALIDACIÓN ---------

def validate_checkpoint_compatibility(agent: DQNAgent, checkpoint_path: str) -> bool:
    """
    Validación básica:
    - input layer compatible con state_dim
    - output layer compatible con action_dim
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Heurística: buscar primer y último Linear
    weight_keys = [k for k in state_dict.keys() if k.endswith("weight")]

    if not weight_keys:
        return False

    first_w = state_dict[weight_keys[0]]
    last_w = state_dict[weight_keys[-1]]

    state_dim_ok = first_w.shape[1] == agent.state_dim
    action_dim_ok = last_w.shape[0] == agent.action_dim

    return state_dim_ok and action_dim_ok


# --------- MAIN ---------

def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["evaluation"]
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
        env = make_maze_env(env_cfg)

        # ReplayBuffer dummy (no usado en evaluación)
        replay_buffer = ReplayBuffer(capacity=1)

        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_space_n,
            replay_buffer=replay_buffer,
        )

        ckpt_path = os.path.join(checkpoint_dir, ckpt)

        if not validate_checkpoint_compatibility(agent, ckpt_path):
            print(f"[SKIP] Checkpoint incompatible: {ckpt}")
            continue

        evaluator = EvaluationController(
            env=env,
            agent=agent,
            model_path=ckpt_path,
            num_episodes=eval_cfg["num_episodes"],
            max_steps_per_episode=eval_cfg["max_steps_per_episode"],
            seed=eval_cfg.get("seed"),
        )

        summary = evaluator.run()
        summary["model"] = ckpt
        all_results.append(summary)

    # --------- TABLA CONSOLA ---------

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

    # --------- GUARDAR RESULTADOS ---------

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados guardados en {results_path}\n")


if __name__ == "__main__":
    main()
