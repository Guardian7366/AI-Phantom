import os
import json
import argparse
import numpy as np
from typing import List, Dict


DEFAULT_RESULTS_DIR = "results/runs"


# ---------------------------------------------------------
# CARGA DE RESULTADOS
# ---------------------------------------------------------

def load_results(results_dir: str) -> List[Dict]:
    experiments = []

    if not os.path.exists(results_dir):
        raise RuntimeError(f"Directorio no encontrado: {results_dir}")

    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            path = os.path.join(root, fname)

            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            if "training" not in data or "evaluation" not in data:
                continue

            data.setdefault(
                "experiment_id",
                fname.replace(".json", "")
            )

            experiments.append(data)

    if not experiments:
        raise RuntimeError(
            "No se encontraron experimentos válidos"
        )

    return experiments


# ---------------------------------------------------------
# ANÁLISIS INDIVIDUAL
# ---------------------------------------------------------

def analyze_experiment(exp: Dict) -> Dict:
    training = exp.get("training", {})
    evaluation = exp.get("evaluation", {})

    flags = []

    train_success = float(training.get("best_success_rate", 0.0))
    eval_success = float(evaluation.get("success_rate", 0.0))
    mean_length = float(evaluation.get("mean_length", np.inf))

    episodes = int(training.get("episodes", 0))

    # -------------------------------------------------
    # 1️⃣ Generalization gap real
    # -------------------------------------------------
    gap = train_success - eval_success

    if gap > 0.2:
        flags.append("GENERALIZATION_GAP")

    # -------------------------------------------------
    # 2️⃣ Convergencia sospechosamente rápida
    # (heurística estable)
    # -------------------------------------------------
    if train_success > 0.95 and episodes < 150:
        flags.append("SUSPICIOUSLY_FAST")

    # -------------------------------------------------
    # 3️⃣ Política ineficiente
    # -------------------------------------------------
    optimal_length = evaluation.get("optimal_length", None)

    if optimal_length is not None:
        if mean_length > 1.5 * optimal_length:
            flags.append("INEFFICIENT_POLICY")
    else:
        # fallback robusto
        if eval_success > 0.8 and mean_length > 20:
            flags.append("POTENTIALLY_INEFFICIENT")

    return {
        "experiment_id": exp.get("experiment_id", "unknown"),
        "train_success": train_success,
        "eval_success": eval_success,
        "mean_length_eval": mean_length,
        "episodes": episodes,
        "gap": gap,
        "flags": flags,
    }


# ---------------------------------------------------------
# ANÁLISIS GLOBAL MULTI-SEED
# ---------------------------------------------------------

def analyze_global_stability(reports: List[Dict]) -> Dict:
    eval_scores = [r["eval_success"] for r in reports]

    mean_eval = float(np.mean(eval_scores))
    std_eval = float(np.std(eval_scores))

    flags = []

    # Alta variabilidad entre seeds
    if std_eval > 0.15:
        flags.append("HIGH_SEED_VARIANCE")

    # Éxito promedio bajo
    if mean_eval < 0.7:
        flags.append("LOW_GLOBAL_PERFORMANCE")

    return {
        "mean_eval_success": mean_eval,
        "std_eval_success": std_eval,
        "flags": flags,
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detección avanzada de falsa convergencia"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio con resultados JSON",
    )
    args = parser.parse_args()

    experiments = load_results(args.results_dir)

    reports = [analyze_experiment(exp) for exp in experiments]
    global_report = analyze_global_stability(reports)

    print("\n=== DETECCIÓN DE FALSA CONVERGENCIA ===\n")

    for r in reports:
        status = "OK" if not r["flags"] else " | ".join(r["flags"])
        print(
            f"{r['experiment_id'][:30]:30} | "
            f"Train: {r['train_success']:.2f} | "
            f"Eval: {r['eval_success']:.2f} | "
            f"Gap: {r['gap']:.2f} | "
            f"Eps: {r['episodes']:4d} | "
            f"{status}"
        )

    print("\n--- Global Stability ---")
    print(
        f"Mean Eval Success : {global_report['mean_eval_success']:.3f}"
    )
    print(
        f"Std  Eval Success : {global_report['std_eval_success']:.3f}"
    )

    if global_report["flags"]:
        print("Global Flags       :", " | ".join(global_report["flags"]))
    else:
        print("Global Flags       : OK")


if __name__ == "__main__":
    main()

