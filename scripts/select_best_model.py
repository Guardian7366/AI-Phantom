import os
import json
import shutil
import argparse

DEFAULT_RANKING_FILE = "results/penalized_ranking.json"
DEFAULT_OUTPUT_DIR = "results/best_model"
DEFAULT_OUTPUT_NAME = "best_model.pth"
DEFAULT_CHECKPOINT_DIR = "results/checkpoints"


# ----------------------------
# Utilidades
# ----------------------------

def load_ranking(ranking_file: str):
    if not os.path.exists(ranking_file):
        raise FileNotFoundError(
            f"No existe el archivo de ranking: {ranking_file}"
        )

    with open(ranking_file, "r") as f:
        ranking = json.load(f)

    if not ranking:
        raise RuntimeError("Ranking vacío")

    return ranking


def resolve_checkpoint_path(experiment_id: str) -> str:
    """
    Nueva convención real del proyecto:

    results/checkpoints/<experiment_id>/best_model.pth
    """

    ckpt_path = os.path.join(
        DEFAULT_CHECKPOINT_DIR,
        experiment_id,
        "best_model.pth"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint esperado: {ckpt_path}"
        )

    return ckpt_path


# ----------------------------
# Selección
# ----------------------------

def select_best_experiment(ranking):
    best = ranking[0]

    if best["score"] <= 0.0:
        raise RuntimeError(
            "El mejor modelo tiene score 0.0 — ningún modelo es aceptable"
        )

    return best


def promote_model(best_exp, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    experiment_id = best_exp["experiment_id"]

    ckpt_src = resolve_checkpoint_path(experiment_id)
    ckpt_dst = os.path.join(output_dir, DEFAULT_OUTPUT_NAME)

    shutil.copyfile(ckpt_src, ckpt_dst)

    metadata = {
        "experiment_id": experiment_id,
        "score": best_exp["score"],
        "eval_success": best_exp["eval_success"],
        "mean_reward": best_exp["mean_reward"],
        "mean_length": best_exp["mean_length"],
        "penalties": best_exp["penalties"],
        "source_checkpoint": ckpt_src,
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return ckpt_dst, metadata


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Selección automática del mejor modelo global"
    )
    parser.add_argument(
        "--ranking_file",
        type=str,
        default=DEFAULT_RANKING_FILE,
        help="Archivo penalized_ranking.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio destino del modelo ganador",
    )
    args = parser.parse_args()

    ranking = load_ranking(args.ranking_file)
    best = select_best_experiment(ranking)

    ckpt_dst, meta = promote_model(best, args.output_dir)

    print("\n=== MODELO GLOBAL SELECCIONADO ===\n")
    print(f"Experiment ID : {meta['experiment_id']}")
    print(f"Score         : {meta['score']}")
    print(f"Eval Success  : {meta['eval_success']}")
    print(f"Mean Reward   : {meta['mean_reward']}")
    print(f"Penalties     : {meta['penalties'] or 'NONE'}")
    print(f"\nCheckpoint promovido a:\n{ckpt_dst}\n")


if __name__ == "__main__":
    main()
