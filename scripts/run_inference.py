import argparse
import yaml

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import PrioritizedReplayBuffer
from controllers.inference_controller import InferenceController


# -------------------------------------------------
# Config
# -------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Builders (robustos y desacoplados)
# -------------------------------------------------

def build_environment(config: dict):
    env_config = config["environment"]
    return MazeEnvironment(config=env_config)


def build_agent(config: dict, env):
    agent_cfg = config.get("agent", {})

    # 🔹 En inferencia no necesitamos replay buffer real
    # Si existe replay_buffer_size (por compatibilidad con train), usarlo.
    buffer_capacity = agent_cfg.get("replay_buffer_size", 1)

    replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space_n,
        replay_buffer=replay_buffer,
        gamma=agent_cfg.get("gamma", 0.99),
        lr=agent_cfg.get("learning_rate", 1e-3),
        batch_size=agent_cfg.get("batch_size", 64),
        min_replay_size=agent_cfg.get("min_replay_size", 1),
        update_frequency=agent_cfg.get("update_frequency", 4),
    )


    return agent


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
    # Load config
    # ----------------------------

    cfg = load_config(args.config)

    env = build_environment(cfg)
    agent = build_agent(cfg, env)

    inference_cfg = cfg.get("inference", {})

    num_episodes = (
        args.episodes
        if args.episodes is not None
        else inference_cfg.get("num_episodes", 10)
    )

    max_steps = inference_cfg.get("max_steps_per_episode", 500)

    # 🔹 Compatibilidad con tu YAML actual (model.path)
    model_path = None
    if "model" in cfg:
        model_path = cfg["model"].get("path")

    controller = InferenceController(
        env=env,
        agent=agent,
        model_path=model_path,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        render=args.render,
        render_delay=inference_cfg.get("render_delay", 0.0),
    )

    results = controller.run()

    print("\n=== INFERENCE SUMMARY ===")
    print(f"Model path     : {results['model_path']}")
    print(f"Episodes       : {results['episodes']}")
    print(f"Success rate   : {results['success_rate']:.3f}")
    print(f"Mean reward    : {results['mean_reward']:.3f}")
    print(f"Mean length    : {results['mean_length']:.2f}")


if __name__ == "__main__":
    main()
