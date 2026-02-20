import argparse
import os
import json
from glob import glob


def load_run_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score(run_payload: dict):
    # Usa lo que guardamos en results/runs/<run_id>.json
    last_eval = run_payload.get("last_eval", {}) or {}
    sr = float(last_eval.get("success_rate", 0.0))
    ratio = float(last_eval.get("mean_ratio_vs_bfs", 999.0))
    steps = float(last_eval.get("mean_steps", 1e9))

    # Orden: success_rate alto, ratio bajo, steps bajos
    return (sr, -ratio, -steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="results/runs")
    ap.add_argument("--out", type=str, default="results/runs/best_run_selected.json")
    args = ap.parse_args()

    paths = sorted(glob(os.path.join(args.runs_dir, "*.json")))
    paths = [p for p in paths if not p.endswith("_history.json")]

    best = None
    best_path = None

    for p in paths:
        try:
            payload = load_run_json(p)
        except Exception:
            continue

        if best is None or score(payload) > score(best):
            best = payload
            best_path = p

    if best is None:
        raise RuntimeError(f"No se encontraron runs válidos en {args.runs_dir}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_payload = {
        "selected_run_json": best_path,
        "run_id": best.get("run_id"),
        "best_eval_success_rate": best.get("best_eval_success_rate"),
        "last_eval": best.get("last_eval"),
        "curriculum_level": best.get("curriculum_level"),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(out_payload, indent=2, ensure_ascii=False))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
