import os
import json
import argparse
import numpy as np
from typing import List, Dict


DEFAULT_RESULTS_DIR = "results"


# -------------------------------------------------
# Carga robusta de experimentos
# -------------------------------------------------

def load_results(results_dir: str) -> List[Dict]:
    experiments = []

    search_dirs = [
        results_dir,
        os.path.join(results_dir, "runs"),  # 🔑 aquí viven los experimentos reales
    ]

    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue

        for fname in os.listdir(base_dir):
            if not fname.endswith(".json"):
                continue

            path = os.path.join(base_dir, fname)

            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            # Solo dicts
            if not isinstance(data, dict):
                continue

            # Debe tener señales de experimento
            if "training" not in data and "evaluation" not in data:
                continue

            # ID del experimento
            data.setdefault(
                "experiment_id",
                os.path.splitext(fname)[0]
            )

            experiments.append(data)

    if not experiments:
        raise RuntimeError(
            "No se encontraron experimentos válidos para analizar falsa convergencia.\n"
            "Esperado: JSONs con claves 'training' y/o 'evaluation' en results/ o results/runs/"
        )

    return experiments


# -------------------------------------------------
# Análisis
# -------------------------------------------------

def analyze_experiment(exp: Dict) -> Dict:
    training = exp.get("training", {})
    evaluation = exp.get("evaluation", {})

    flags = []

    train_success = training.get("best_success_rate", 0.0)
    eval_success = evaluation.get("success_rate", 0.0)

    mean_length = evaluation.get("mean_length", np.inf)
    episodes = training.get("episodes", np.inf)

    # 1. Éxito alto en training, bajo en evaluación
    if train_success > 0.9 and eval_success < 0.7:
        flags.append("OVERFIT_TRAINING")

    # 2. Convergencia sospechosamente rápida
    if train_success > 0.9 and episodes < 0.3 * training.get("episodes", episodes):
        flags.append("TOO_FAST_CONVERGENCE")

    # 3. Política ineficiente
    optimal_len = evaluation.get("optimal_length", mean_length)
    if eval_success > 0.8 and mean_length > 1.5 * optimal_len:
        flags.append("INEFFICIENT_POLICY")

    return {
        "experiment_id": exp.get("experiment_id", "unknown"),
        "train_success": train_success,
        "eval_success": eval_success,
        "mean_length_eval": mean_length,
        "flags": flags,
    }


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detección automática de falsa convergencia"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio base de resultados",
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
