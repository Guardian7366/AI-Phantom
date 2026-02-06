import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


DEFAULT_RESULTS_DIR = "results"
OUTPUT_DIR = "results/plots"


def load_results(results_dir: str) -> List[Dict]:
    experiments = []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(results_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)

        experiments.append(data)

    if not experiments:
        raise RuntimeError("No se encontraron archivos de resultados")

    return experiments


def extract_metrics(experiments: List[Dict]) -> Dict[str, List[float]]:
    success_eval = []
    reward_eval = []
    length_eval = []

    labels = []

    for exp in experiments:
        evaluation = exp.get("evaluation", {})
        success_eval.append(evaluation.get("success_rate", 0.0))
        reward_eval.append(evaluation.get("mean_reward", 0.0))
        length_eval.append(evaluation.get("mean_length", 0.0))
        labels.append(exp.get("experiment_id", "unknown"))

    return {
        "labels": labels,
        "success_eval": success_eval,
        "reward_eval": reward_eval,
        "length_eval": length_eval,
    }


def plot_boxplots(metrics: Dict[str, List[float]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    data = [
        metrics["success_eval"],
        metrics["reward_eval"],
        metrics["length_eval"],
    ]

    titles = [
        "Success Rate (Evaluation)",
        "Mean Reward (Evaluation)",
        "Mean Episode Length (Evaluation)",
    ]

    plt.figure()
    plt.boxplot(data, labels=titles)
    plt.title("Distribución de métricas de evaluación")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplots.png"))
    plt.close()


def plot_success_curve(metrics: Dict[str, List[float]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    success = np.array(metrics["success_eval"])
    order = np.argsort(success)[::-1]
    sorted_success = success[order]

    plt.figure()
    plt.plot(sorted_success)
    plt.xlabel("Experimentos (ordenados)")
    plt.ylabel("Success Rate")
    plt.title("Success Rate por experimento (evaluación)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_curve.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualización de resultados AI Phantom"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio con results/*.json",
    )
    args = parser.parse_args()

    experiments = load_results(args.results_dir)
    metrics = extract_metrics(experiments)

    plot_boxplots(metrics, OUTPUT_DIR)
    plot_success_curve(metrics, OUTPUT_DIR)

    print(f"Gráficas guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
