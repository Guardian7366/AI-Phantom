import argparse
import yaml

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from controllers.inference_controller import InferenceController


# -------------------------------------------------
# Factories
# -------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_environment(cfg: dict):
    env_cfg = cfg["environment"]
    return MazeEnvironment(**env_cfg)


def make_agent(cfg: dict, state_dim: int, action_dim: int):
    agent_cfg = cfg["agent"]
    return DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        **agent_cfg
    )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with best trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/maze_inference.yaml",
        help="Inference config file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of inference episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment"
    )
    args = parser.parse_args()

    # ----------------------------
    # Configuración
    # ----------------------------

    cfg = load_config(args.config)

    env = make_environment(cfg)

    agent = make_agent(
        cfg,
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )

    controller_cfg = cfg.get("inference", {})

    num_episodes = (
        args.episodes
        if args.episodes is not None
        else controller_cfg.get("num_episodes", 10)
    )

    max_steps = controller_cfg.get("max_steps_per_episode", 500)

    # ----------------------------
    # Inferencia
    # ----------------------------

    controller = InferenceController(
        env=env,
        agent=agent,
        model_path=controller_cfg.get("model_path"),  # puede ser None
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        render=args.render,
        render_delay=controller_cfg.get("render_delay", 0.0),
    )

    results = controller.run()

    # ----------------------------
    # Output
    # ----------------------------

    print("\n=== INFERENCE SUMMARY ===")
    print(f"Model path     : {results['model_path']}")
    print(f"Episodes       : {results['episodes']}")
    print(f"Success rate   : {results['success_rate']:.3f}")
    print(f"Mean reward    : {results['mean_reward']:.3f}")
    print(f"Mean length    : {results['mean_length']:.2f}")


if __name__ == "__main__":
    main()
