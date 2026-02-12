import argparse
import yaml
import numpy as np

from scripts.run_inference import build_environment, build_agent
from agents.dqn.dqn_agent import DQNAgent


# -------------------------------------------------
# Config
# -------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Core evaluation logic
# -------------------------------------------------

def run_episodes(env, agent: DQNAgent, num_episodes: int, max_steps: int, epsilon: float):
    """
    Ejecuta múltiples episodios forzando un epsilon específico.
    No usa controller para mantener control total del ruido.
    """
    successes = []
    rewards = []
    lengths = []

    agent.set_mode(training=False)

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        success = False

        while not done and steps < max_steps:
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state
            steps += 1

            if info.get("success", False):
                success = True

        successes.append(1 if success else 0)
        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robustness under action perturbation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/maze_inference.yaml",
        help="Inference config file",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per condition",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Artificial epsilon for perturbation test",
    )

    args = parser.parse_args()

    # ----------------------------
    # Load config
    # ----------------------------

    cfg = load_config(args.config)

    env = build_environment(cfg)
    agent = build_agent(cfg, env)

    # Load model
    model_path = cfg.get("model", {}).get("path")
    if model_path is None:
        raise RuntimeError("Model path not found in config")

    agent.load(model_path)

    # Factory isolation (muy importante)
    base_agent = agent.factory()
    base_agent.load(model_path)

    perturbed_agent = agent.factory()
    perturbed_agent.load(model_path)

    max_steps = cfg.get("inference", {}).get("max_steps_per_episode", 500)

    print("\n=== PERTURBED EVALUATION ===\n")

    # ----------------------------
    # Baseline
    # ----------------------------

    baseline_results = run_episodes(
        env=env.factory(),
        agent=base_agent,
        num_episodes=args.episodes,
        max_steps=max_steps,
        epsilon=0.0,
    )

    print("Baseline (epsilon=0.0)")
    print(baseline_results)

    # ----------------------------
    # Perturbed
    # ----------------------------

    perturbed_results = run_episodes(
        env=env.factory(),
        agent=perturbed_agent,
        num_episodes=args.episodes,
        max_steps=max_steps,
        epsilon=args.epsilon,
    )

    print(f"\nPerturbed (epsilon={args.epsilon})")
    print(perturbed_results)

    # ----------------------------
    # Degradation analysis
    # ----------------------------

    success_drop = baseline_results["success_rate"] - perturbed_results["success_rate"]
    length_increase = perturbed_results["mean_length"] - baseline_results["mean_length"]

    print("\n--- Degradation ---")
    print(f"Success Drop  : {success_drop:.4f}")
    print(f"Length Change : {length_increase:.4f}")

    if success_drop < 0.02:
        robustness_flag = "ROBUST_POLICY"
    elif success_drop < 0.10:
        robustness_flag = "MODERATELY_ROBUST"
    else:
        robustness_flag = "FRAGILE_POLICY"

    print(f"\nRobustness Classification: {robustness_flag}")


if __name__ == "__main__":
    main()
