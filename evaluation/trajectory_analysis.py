import os
import json
import numpy as np
from typing import List, Dict


# =================================================
# Trajectory collection (USADO por visualize_trajectories)
# =================================================

def collect_trajectories(env, agent, episodes: int = 5, max_steps: int = 500):
    """
    Ejecuta episodios en modo inferencia y guarda trayectorias.
    """
    trajectories = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        step = 0

        episode_data = {
            "episode": ep,
            "positions": [tuple(env.agent_pos)],
            "rewards": [],
            "success": False,
        }

        while not done and step < max_steps:
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)

            episode_data["positions"].append(tuple(env.agent_pos))
            episode_data["rewards"].append(reward)

            if info.get("success", False):
                episode_data["success"] = True

            state = next_state
            step += 1

        trajectories.append(episode_data)

    return trajectories


def save_trajectories(trajectories: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)


# =================================================
# Result analysis (USADO por compare_results)
# =================================================

DEFAULT_RESULTS_DIR = "results"


def is_experiment_file(data) -> bool:
    return isinstance(data, dict) and (
        "training" in data or "evaluation" in data
    )


def load_results(results_dir: str) -> List[Dict]:
    experiments = []

    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            path = os.path.join(root, fname)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue

            if not is_experiment_file(data):
                continue

            experiments.append(data)

    if not experiments:
        raise RuntimeError("No se encontraron archivos de resultados válidos")

    return experiments


def summarize_experiments(experiments: List[Dict]) -> List[Dict]:
    summary = []

    for exp in experiments:
        training = exp.get("training", {})
        evaluation = exp.get("evaluation", {})

        summary.append({
            "experiment_id": exp.get("experiment_id", "unknown"),
            "episodes": training.get("episodes"),
            "best_success_rate": training.get("best_success_rate"),
            "mean_reward_eval": evaluation.get("mean_reward"),
            "mean_length_eval": evaluation.get("mean_length"),
            "success_rate_eval": evaluation.get("success_rate"),
        })

    return summary
