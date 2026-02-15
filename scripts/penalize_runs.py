import os
import json
import argparse
import numpy as np
from typing import List, Dict

DEFAULT_RESULTS_DIR = "results/runs"
OUTPUT_FILE = "results/penalized_ranking.json"


# ----------------------------
# Carga de resultados (recursiva)
# ----------------------------

def load_results(results_dir: str) -> List[Dict]:
    experiments = []

    for root, _, files in os.walk(results_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            full_path = os.path.join(root, fname)

            try:
                with open(full_path, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            # SOLO aceptar JSONs de experimentos
            if not isinstance(data, dict):
                continue

            if "experiment_id" not in data:
                continue

            experiments.append(data)

    if not experiments:
        raise RuntimeError(
            f"No se encontraron resultados válidos en {results_dir}"
        )

    return experiments


# ----------------------------
# Penalización y scoring
# ----------------------------

def compute_penalty(exp: Dict) -> Dict:
    training = exp.get("training", {})
    evaluation = exp.get("evaluation", {})

    train_success = training.get("best_success_rate", 0.0)
    eval_success = evaluation.get("success_rate", 0.0)
    mean_length = evaluation.get("mean_length", np.inf)
    mean_reward = evaluation.get("mean_reward", 0.0)

    penalties = []
    score = eval_success

    # 1. Overfitting
    if train_success > 0.9 and eval_success < 0.7:
        penalties.append("OVERFIT")
        score -= 0.3

    # 2. Política lenta (si resuelve pero tarda demasiado)
    if eval_success > 0.7 and mean_length > 1.8 * 8:
        penalties.append("INEFFICIENT")
        score -= 0.2

    # 3. Reward hack
    if mean_reward > 0 and eval_success < 0.5:
        penalties.append("REWARD_HACK")
        score -= 0.2

    score = max(score, 0.0)

    return {
        "experiment_id": exp.get("experiment_id", "unknown"),
        "score": round(score, 4),
        "eval_success": round(eval_success, 4),
        "mean_reward": round(mean_reward, 2),
        "mean_length": round(mean_length, 2),
        "penalties": penalties,
    }



def rank_experiments(experiments: List[Dict]) -> List[Dict]:
    ranked = [compute_penalty(exp) for exp in experiments]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Penalización automática de runs defectuosos"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directorio con runs de experimentos",
    )
    args = parser.parse_args()

    experiments = load_results(args.results_dir)
    ranking = rank_experiments(experiments)

    print("\n=== RANKING PENALIZADO DE RUNS ===\n")

    for idx, r in enumerate(ranking):
        penalty_tag = " | ".join(r["penalties"]) if r["penalties"] else "OK"
        print(
            f"{idx+1:2d}. {r['experiment_id'][:30]:30} | "
            f"Score: {r['score']:.2f} | "
            f"Succ: {r['eval_success']:.2f} | "
            f"Len: {r['mean_length']:.1f} | "
            f"{penalty_tag}"
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(ranking, f, indent=2)

    print(f"\nRanking guardado en {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
