# scripts/evaluate_checkpoints.py
# Evaluación CONSISTENTE con entrenamiento (Opción A)

import os
import json
import yaml
import torch
import argparse

from controllers.evaluation_controller import EvaluationController
from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import PrioritizedReplayBuffer


# -------------------------------------------------
# ARGPARSE
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/maze_train.yaml",  # 🔥 USAMOS TRAIN CONFIG
        help="Ruta al archivo YAML de entrenamiento"
    )
    return parser.parse_args()


# -------------------------------------------------
# CARGA SEGURA DE CHECKPOINT
# -------------------------------------------------

def validate_checkpoint_compatibility(agent: DQNAgent, checkpoint_path: str) -> bool:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        agent.policy_net.load_state_dict(state_dict, strict=True)
        return True

    except Exception as e:
        print(f"[SKIP] Incompatible ({e}): {checkpoint_path}")
        return False


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    env_cfg = config["environment"]
    training_cfg = config["training"]

    checkpoint_root = training_cfg["checkpoint_dir"]
    results_root = training_cfg["results_dir"]

    all_results = []

    # -------------------------------------------------
    # Buscar checkpoints BEST por experimento
    # -------------------------------------------------

    checkpoint_files = []

    for root, _, files in os.walk(checkpoint_root):
        for f in files:
            if f == "best_model.pth":
                checkpoint_files.append(os.path.join(root, f))

    checkpoint_files = sorted(checkpoint_files)

    if not checkpoint_files:
        raise RuntimeError(
            f"No se encontraron best_model.pth en {checkpoint_root}"
        )

    # -------------------------------------------------
    # Loop evaluación
    # -------------------------------------------------

    for ckpt_path in checkpoint_files:

        experiment_id = os.path.basename(os.path.dirname(ckpt_path))
        result_json_path = os.path.join(results_root, f"{experiment_id}.json")

        if not os.path.exists(result_json_path):
            print(f"[SKIP] No existe JSON de experimento: {experiment_id}")
            continue

        with open(result_json_path, "r", encoding="utf-8") as f:
            experiment_data = json.load(f)

        final_level = experiment_data["training"].get(
            "final_curriculum_level", 0
        )

        # -------------------------------------------------
        # ENV FACTORY (idéntica a TrainingController)
        # -------------------------------------------------

        def env_factory():
            env = MazeEnvironment(env_cfg)
            env.set_curriculum_level(final_level)

            # 🔥 sincronizar max_steps EXACTAMENTE
            env.max_steps = training_cfg["max_steps_per_episode"]

            return env

        # -------------------------------------------------
        # AGENT FACTORY
        # -------------------------------------------------

        temp_env = MazeEnvironment(env_cfg)

        def agent_factory():
            replay_buffer = PrioritizedReplayBuffer(capacity=1)
            agent = DQNAgent(
                state_dim=temp_env.state_dim,
                action_dim=temp_env.action_space_n,
                replay_buffer=replay_buffer,
                gamma=config["agent"]["gamma"],
                lr=config["agent"]["learning_rate"],
                batch_size=config["agent"]["batch_size"],
                min_replay_size=config["agent"]["min_replay_size"],
                tau=config["agent"]["tau"],
                update_frequency=config["agent"]["update_frequency"],
            )
            agent.set_mode(False)
            return agent

        # Validación
        temp_agent = agent_factory()
        if not validate_checkpoint_compatibility(temp_agent, ckpt_path):
            continue

        # -------------------------------------------------
        # Evaluador
        # -------------------------------------------------

        evaluator = EvaluationController(
            env_factory=env_factory,
            agent_factory=agent_factory,
            config={
                "evaluation": {
                    "num_episodes": 200,
                    "seed": 123,
                }
            },
        )

        summary = evaluator.evaluate_checkpoint(
            ckpt_path,
            forced_curriculum_level=final_level,
        )

        summary["model"] = experiment_id
        summary["curriculum_level"] = final_level

        all_results.append(summary)

    # -------------------------------------------------
    # TABLA
    # -------------------------------------------------

    print("\n=== COMPARATIVA CONSISTENTE (TRAIN CONFIG) ===\n")

    header = (
        f"{'Model':35} | {'Lvl':4} | "
        f"{'Success':8} | {'MeanLen':8} | {'MeanRew':8}"
    )
    print(header)
    print("-" * len(header))

    for r in all_results:
        print(
            f"{r['model']:35} | "
            f"{r['curriculum_level']:4} | "
            f"{r['success_rate']:.3f}   | "
            f"{r['mean_length']:.2f}   | "
            f"{r['mean_reward']:.2f}"
        )

    # Guardar
    output_path = "results/checkpoint_evaluation_consistent.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados guardados en {output_path}\n")


if __name__ == "__main__":
    main()
