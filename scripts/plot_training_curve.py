# scripts/plot_training_curve.py

import os
import json
import argparse
import glob
import matplotlib.pyplot as plt


# ============================================================
# Utilidades
# ============================================================

def load_run_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_runs(runs_dir):
    pattern = os.path.join(runs_dir, "*.json")
    return sorted(glob.glob(pattern))


# ============================================================
# Plot principal
# ============================================================

def plot_training_curves(run_data, experiment_id):
    training = run_data["training"]

    reward_history = training.get("reward_history", [])
    length_history = training.get("length_history", [])
    success_history = training.get("success_history", [])
    loss_history = training.get("loss_history", [])

    episodes = range(1, len(reward_history) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Training Curves - {experiment_id}", fontsize=14)

    # Reward
    axes[0, 0].plot(episodes, reward_history)
    axes[0, 0].set_title("Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)

    # Length
    axes[0, 1].plot(episodes, length_history)
    axes[0, 1].set_title("Episode Length")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].grid(True)

    # Success
    axes[1, 0].plot(episodes, success_history)
    axes[1, 0].set_title("Success (1/0)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Success")
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].grid(True)

    # Loss
    axes[1, 1].plot(episodes, loss_history)
    axes[1, 1].set_title("Loss per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot training curves from run JSON.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Exact experiment_id (ej: maze_dqn_v1_seed42)"
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="results/runs",
        help="Directory containing run JSON files"
    )

    args = parser.parse_args()

    if not os.path.exists(args.runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {args.runs_dir}")

    available_runs = get_available_runs(args.runs_dir)

    if len(available_runs) == 0:
        raise RuntimeError("No run JSON files found.")

    # Si no especifica run → usar el primero
    if args.run is None:
        selected_path = available_runs[0]
        print(f"[INFO] No run specified. Using: {os.path.basename(selected_path)}")
    else:
        matches = [
            path for path in available_runs
            if args.run in os.path.basename(path)
        ]

        if len(matches) == 0:
            raise RuntimeError(f"Run '{args.run}' not found in {args.runs_dir}")

        selected_path = matches[0]

    run_data = load_run_json(selected_path)
    experiment_id = run_data.get("experiment_id", "unknown_experiment")

    print(f"[INFO] Plotting training curves for: {experiment_id}")

    plot_training_curves(run_data, experiment_id)


if __name__ == "__main__":
    main()
