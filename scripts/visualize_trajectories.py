import os

from controllers.inference_controller import InferenceController, DEFAULT_BEST_MODEL_PATH
from evaluation.trajectory_analysis import collect_trajectories, save_trajectories

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import ReplayBuffer
from utils.seeding import set_global_seed
from utils.logging import load_yaml_config


def main():
    print("\n=== Visualización de Trayectorias ===")

    # ----------------------------
    # Config
    # ----------------------------
    config = load_yaml_config("configs/maze_inference.yaml")
    set_global_seed(config.get("seed", 123))

    # ----------------------------
    # Environment
    # ----------------------------
    env = MazeEnvironment(config)

    # ----------------------------
    # Agent
    # ----------------------------
    state_dim = env.observation_space
    action_dim = env.action_space

    raw_agent_cfg = dict(config.get("agent", {}))

    # ✅ Parámetros permitidos por DQNAgent.__init__
    allowed_keys = {
        "gamma",
        "lr",
        "batch_size",
        "target_update_freq",
        "device",
    }

    agent_cfg = {
        k: v for k, v in raw_agent_cfg.items() if k in allowed_keys
    }

    replay_buffer = ReplayBuffer(capacity=1)  # dummy buffer
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        **agent_cfg,
    )

    # ----------------------------
    # Controller
    # ----------------------------
    controller = InferenceController(
        env=env,
        agent=agent,
    )

    # ----------------------------
    # Trajectories
    # ----------------------------
    trajectories = collect_trajectories(
        env=controller.env,
        agent=controller.agent,
        episodes=5,
    )

    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "maze_trajectories.json")
    save_trajectories(trajectories, output_path)

    print(f"[OK] Trayectorias guardadas en: {output_path}")

    for t in trajectories:
        print(
            f"Ep {t['episode']} | Steps: {len(t['positions'])} | Success: {t['success']}"
        )


if __name__ == "__main__":
    main()

