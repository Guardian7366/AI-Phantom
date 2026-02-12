import argparse
import yaml
import numpy as np

from scripts.run_inference import build_environment, build_agent
from environments.maze.stochastic_wrapper import StochasticMazeWrapper


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_eval_agent(cfg, env, model_path):
    """
    Construye un agente NUEVO para evaluación
    y carga correctamente los pesos.
    """
    agent = build_agent(cfg, env)
    agent.load(model_path)
    agent.set_mode(training=False)
    return agent


def run_episodes(env, agent, num_episodes, max_steps):
    successes = []
    rewards = []
    lengths = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done and steps < max_steps:
            action = agent.select_action(state, epsilon=0.0)
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate under stochastic environment dynamics"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/maze_inference.yaml",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Transition noise probability",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    base_env = build_environment(cfg)

    model_path = cfg.get("model", {}).get("path")
    if model_path is None:
        raise RuntimeError("Model path not found in config")

    clean_env = base_env.factory()
    noisy_env = StochasticMazeWrapper(
        base_env.factory(),
        action_noise_prob=args.noise,
    )

    max_steps = cfg.get("inference", {}).get("max_steps_per_episode", 500)

    print("\n=== STOCHASTIC ENVIRONMENT EVALUATION ===\n")

    clean_agent = build_eval_agent(cfg, clean_env, model_path)
    clean_results = run_episodes(
        clean_env,
        clean_agent,
        args.episodes,
        max_steps,
    )

    print("Clean Environment")
    print(clean_results)

    noisy_agent = build_eval_agent(cfg, noisy_env, model_path)
    noisy_results = run_episodes(
        noisy_env,
        noisy_agent,
        args.episodes,
        max_steps,
    )

    print(f"\nNoisy Environment (noise={args.noise})")
    print(noisy_results)

    success_drop = clean_results["success_rate"] - noisy_results["success_rate"]

    print("\n--- Structural Robustness ---")
    print(f"Success Drop : {success_drop:.4f}")

    if success_drop < 0.05:
        classification = "STRUCTURALLY_ROBUST"
    elif success_drop < 0.20:
        classification = "MODERATE_SENSITIVITY"
    else:
        classification = "FRAGILE_TO_DYNAMICS"

    print(f"Classification : {classification}")


if __name__ == "__main__":
    main()
