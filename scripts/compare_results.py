import os
import json
import argparse
import numpy as np
from typing import List, Dict


DEFAULT_RESULTS_DIR = "results"


def is_experiment_file(data) -> bool:
    """
    Verifica si el JSON corresponde a un experimento entrenado
    y no a trayectorias, plots u otros artefactos.
    """
    if not isinstance(data, dict):
        return False

    # Debe tener al menos alguna de estas secciones
    return "training" in data or "evaluation" in data


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
        raise RuntimeError(
            "No se encontraron archivos de resultados de experimentos válidos"
        )

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


def print_table(summary: List[Dict], sort_by: str):
    summary = sorted(
        summary,
        key=lambda x: x.get(sort_by, -np.inf),
        reverse=True,
    )

    header = (
        f"{'Experiment':30} | "
        f"{'Succ(eval)':10} | "
        f"{'Rew(eval)':10} | "
        f"{'Len(eval)':10} | "
        f"{'Succ(train)':10} | "
        f"{'Episodes':8}"
    )

    print("\n=== COMPARATIVA DE EXPERIMENTOS ===\n")
    print(header)
    print("-" * len(header))

    for row in summary:
        print(
            f"{row['experiment_id'][:30]:30} | "
            f"{row['success_rate_eval']:.3f}      | "
            f"{row['mean_reward_eval']:.2f}      | "
            f"{row['mean_length_eval']:.2f}      | "
            f"{row['best_success_rate']:.3f}      | "
            f"{row['episodes']:8}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Comparador de resultados AI Phantom"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio con archivos results/**/*.json",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="success_rate_eval",
        choices=[
            "success_rate_eval",
            "mean_reward_eval",
            "mean_length_eval",
        ],
        help="Métrica principal para ordenar",
    )
    args = parser.parse_args()

    experiments = load_results(args.results_dir)
    summary = summarize_experiments(experiments)
    print_table(summary, args.sort_by)


if __name__ == "__main__":
    main()
