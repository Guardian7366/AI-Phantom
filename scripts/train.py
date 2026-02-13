import argparse
import yaml
import torch

from controllers.training_controller import TrainingController
from utils.seeding import set_global_seed

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import ReplayBuffer


# -------------------------------------------------
# Config
# -------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



# -------------------------------------------------
# Builders
# -------------------------------------------------
def build_environment(config: dict):
    env_config = config["environment"]
    return MazeEnvironment(config=env_config)


def build_agent(config: dict, env):
    agent_cfg = config["agent"]

    replay_buffer = ReplayBuffer(
        capacity=agent_cfg["replay_buffer_size"]
    )

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space_n,
        replay_buffer=replay_buffer,
        gamma=agent_cfg.get("gamma", 0.99),
        lr=agent_cfg.get("learning_rate", 1e-3),
        batch_size=agent_cfg.get("batch_size", 64),
        min_replay_size=agent_cfg.get("min_replay_size", 1000),
        tau=agent_cfg.get("tau", 0.005),
        update_frequency=agent_cfg.get("update_frequency", 4),
    )

    return agent


# -------------------------------------------------
# Experiment
# -------------------------------------------------

def run_experiment(config: dict, seed: int, experiment_id: str):
    set_global_seed(seed)

    env = build_environment(config)
    agent = build_agent(config, env)

    controller = TrainingController(
        env=env,
        agent=agent,
        num_episodes=config["training"]["num_episodes"],
        max_steps_per_episode=config["training"]["max_steps_per_episode"],
        epsilon_start=config["training"]["epsilon_start"],
        epsilon_end=config["training"]["epsilon_end"],
        epsilon_decay_episodes=config["training"]["epsilon_decay_episodes"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        results_dir=config["training"]["results_dir"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        success_threshold=config["training"]["success_threshold"],
        success_window=config["training"]["success_window"],
        experiment_id=experiment_id,
    )

    return controller.train()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Phantom Training Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/maze_train.yaml",
        help="Ruta al archivo de configuración YAML",
    )
    args = parser.parse_args()

    # -----------------------------------
    # CUDA Optimizations
    # -----------------------------------
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = load_config(args.config)

    seeds = config.get("seeds", [42])
    experiment_name = config.get("experiment_name", "maze_experiment")

    for idx, seed in enumerate(seeds):
        experiment_id = f"{experiment_name}_seed{seed}"

        print(f"\n=== Experimento {idx + 1}/{len(seeds)} | Seed {seed} ===")
        print(f"ID: {experiment_id}\n")

        run_experiment(
            config=config,
            seed=seed,
            experiment_id=experiment_id,
        )


if __name__ == "__main__":
    main()
