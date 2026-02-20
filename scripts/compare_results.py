import argparse
import os
import json
from glob import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="results/runs")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    paths = sorted(glob(os.path.join(args.runs_dir, "*.json")))
    paths = [p for p in paths if not p.endswith("_history.json")]

    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                run = json.load(f)
        except Exception:
            continue

        ev = run.get("last_eval", {}) or {}
        rows.append({
            "path": p,
            "run_id": run.get("run_id"),
            "sr": float(ev.get("success_rate", 0.0)),
            "ratio": float(ev.get("mean_ratio_vs_bfs", 999.0)),
            "steps": float(ev.get("mean_steps", 1e9)),
            "curriculum": run.get("curriculum_level"),
        })

    rows.sort(key=lambda x: (x["sr"], -x["ratio"], -x["steps"]), reverse=True)

    print(f"Found runs: {len(rows)}")
    print(f"Top {min(args.topk, len(rows))}:")
    for r in rows[: args.topk]:
        print(
            f"- {r['run_id']} | SR={r['sr']:.3f} ratio={r['ratio']:.3f} steps={r['steps']:.1f} "
            f"lvl={r['curriculum']} | {os.path.basename(r['path'])}"
        )


if __name__ == "__main__":
    main()
