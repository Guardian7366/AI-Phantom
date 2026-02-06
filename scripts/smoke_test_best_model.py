import sys
import yaml
import numpy as np

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from controllers.inference_controller import InferenceController


# -------------------------------------------------
# Utils
# -------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def fail(msg: str):
    print(f"[SMOKE TEST ❌] {msg}")
    sys.exit(1)


def success(msg: str):
    print(f"[SMOKE TEST ✅] {msg}")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    config_path = "configs/maze_inference.yaml"

    print("=== Smoke Test: Best Model ===")

    # ----------------------------
    # Load config
    # ----------------------------

    try:
        cfg = load_config(config_path)
        success("Config loaded")
    except Exception as e:
        fail(f"Failed to load config: {e}")

    # ----------------------------
    # Environment
    # ----------------------------

    try:
        env = MazeEnvironment(**cfg["environment"])
        success("Environment initialized")
    except Exception as e:
        fail(f"Environment failed to initialize: {e}")

    # ----------------------------
    # Agent
    # ----------------------------

    try:
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            **cfg["agent"]
        )
        success("Agent initialized")
    except Exception as e:
        fail(f"Agent failed to initialize: {e}")

    # ----------------------------
    # Inference controller
    # ----------------------------

    inference_cfg = cfg.get("inference", {})
    model_path = inference_cfg.get("model_path")

    if model_path is None:
        fail("model_path not defined in inference config")

    controller = InferenceController(
        env=env,
        agent=agent,
        model_path=model_path,
        num_episodes=5,
        max_steps_per_episode=inference_cfg.get("max_steps_per_episode", 500),
        render=False,
    )

    # ----------------------------
    # Run inference
    # ----------------------------

    try:
        results = controller.run()
        success("Inference executed")
    except Exception as e:
        fail(f"Inference crashed: {e}")

    # ----------------------------
    # Sanity checks
    # ----------------------------

    if np.isnan(results["mean_reward"]):
        fail("Mean reward is NaN")

    if results["mean_length"] <= 0:
        fail("Invalid episode length")

    if results["episodes"] == 0:
        fail("No episodes executed")

    success(f"Success rate: {results['success_rate']:.2f}")
    success(f"Mean reward: {results['mean_reward']:.2f}")
    success(f"Mean length: {results['mean_length']:.2f}")

    print("\n[SMOKE TEST PASSED 🎉] Best model is runnable and sane")


if __name__ == "__main__":
    main()
