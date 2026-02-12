import os
import json
import argparse
import numpy as np
from typing import List, Dict


DEFAULT_RESULTS_DIR = "results/runs"


def load_experiments(results_dir: str) -> List[Dict]:
    experiments = []

    if not os.path.exists(results_dir):
        raise RuntimeError(f"Directorio no encontrado: {results_dir}")

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(results_dir, fname)

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if "training" not in data or "evaluation" not in data:
            continue

        experiments.append(data)

    if not experiments:
        raise RuntimeError("No se encontraron experimentos válidos")

    return experiments


def compute_statistics(values: List[float]) -> Dict:
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Análisis de robustez multi-seed"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio con JSON de runs",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.results_dir)

    train_success = []
    eval_success = []
    train_length = []
    eval_length = []
    epsilons = []

    for exp in experiments:
        training = exp["training"]
        evaluation = exp["evaluation"]

        train_success.append(training.get("best_success_rate", 0.0))
        eval_success.append(evaluation.get("success_rate", 0.0))
        train_length.append(training.get("mean_length", 0.0))
        eval_length.append(evaluation.get("mean_length", 0.0))
        epsilons.append(training.get("final_epsilon", 0.0))

    print("\n=== ROBUSTNESS ANALYSIS ===\n")

    print("Train Success Stats:", compute_statistics(train_success))
    print("Eval  Success Stats:", compute_statistics(eval_success))
    print("Train Length  Stats:", compute_statistics(train_length))
    print("Eval  Length  Stats:", compute_statistics(eval_length))
    print("Final Epsilon Stats:", compute_statistics(epsilons))

    # Global Stability Check
    eval_std = np.std(eval_success)

    if eval_std < 0.01:
        stability_flag = "HIGHLY_STABLE"
    elif eval_std < 0.05:
        stability_flag = "STABLE"
    else:
        stability_flag = "VARIABLE"

    print("\n--- Global Stability ---")
    print(f"Std Eval Success : {eval_std:.4f}")
    print(f"Stability        : {stability_flag}")


if __name__ == "__main__":
    main()
