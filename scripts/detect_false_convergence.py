import os
import json
import argparse
import numpy as np
from typing import List, Dict


DEFAULT_RESULTS_DIR = "results"


def load_results(results_dir: str) -> List[Dict]:
    experiments = []

    for fname in os.listdir(results_dir):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(results_dir, fname), "r") as f:
            experiments.append(json.load(f))

    if not experiments:
        raise RuntimeError("No se encontraron resultados")

    return experiments


def analyze_experiment(exp: Dict) -> Dict:
    training = exp.get("training", {})
    evaluation = exp.get("evaluation", {})

    flags = []

    train_success = training.get("best_success_rate", 0.0)
    eval_success = evaluation.get("success_rate", 0.0)

    mean_length = evaluation.get("mean_length", np.inf)
    episodes = training.get("episodes", np.inf)

    # 1. Éxito alto en entrenamiento, bajo en evaluación
    if train_success > 0.9 and eval_success < 0.7:
        flags.append("OVERFIT_TRAINING")

    # 2. Convergencia sospechosamente rápida
    if train_success > 0.9 and episodes < 0.3 * training.get("episodes", episodes):
        flags.append("TOO_FAST_CONVERGENCE")

    # 3. Reward alto pero episodios largos
    if eval_success > 0.8 and mean_length > 0.7 * mean_length:
        flags.append("INEFFICIENT_POLICY")

    return {
        "experiment_id": exp.get("experiment_id", "unknown"),
        "train_success": train_success,
        "eval_success": eval_success,
        "mean_length_eval": mean_length,
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detección automática de falsa convergencia"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio con results/*.json",
    )
    args = parser.parse_args()

    experiments = load_results(args.results_dir)

    reports = [analyze_experiment(exp) for exp in experiments]

    print("\n=== DETECCIÓN DE FALSA CONVERGENCIA ===\n")

    for r in reports:
        status = "OK" if not r["flags"] else " | ".join(r["flags"])
        print(
            f"{r['experiment_id'][:30]:30} | "
            f"Train: {r['train_success']:.2f} | "
            f"Eval: {r['eval_success']:.2f} | "
            f"Len: {r['mean_length_eval']:.1f} | "
            f"{status}"
        )


if __name__ == "__main__":
    main()
