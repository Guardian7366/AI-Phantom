import os
import json
import yaml

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import ReplayBuffer
from controllers.inference_controller import InferenceController


# -------------------------------------------------
# Utils
# -------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print("\n=== Collect Trajectories: Best Model ===")

    config_path = "configs/maze_inference.yaml"
    output_dir = "results/trajectories"
    output_path = os.path.join(output_dir, "best_model_trajectories.json")

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Load config
    # ----------------------------

    cfg = load_config(config_path)

    # ----------------------------
    # Environment
    # ----------------------------

    env = MazeEnvironment(config=cfg)

    # ----------------------------
    # Agent (inferencia pura)
    # ----------------------------

    raw_agent_cfg = cfg.get("agent", {})
    agent_cfg = {
        k: v
        for k, v in raw_agent_cfg.items()
        if k not in {"type", "epsilon"}
    }

    replay_buffer = ReplayBuffer(capacity=1)

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space_n,
        replay_buffer=replay_buffer,
        **agent_cfg
    )

    agent.set_mode(training=False)

    # ----------------------------
    # Controller
    # ----------------------------

    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path")

    if model_path is None:
        raise RuntimeError("model.path not defined in config")

    controller = InferenceController(
        env=env,
        agent=agent,
        model_path=model_path,
        num_episodes=5,
        max_steps_per_episode=cfg.get("environment", {}).get("max_steps", 500),
        render=False,
    )

    # ----------------------------
    # Run & collect trajectories
    # ----------------------------

    trajectories = []

    for ep in range(controller.num_episodes):
        state = controller.env.reset()
        done = False

        episode_data = {
            "episode": ep,
            "positions": [],
            "rewards": [],
            "success": False,
        }

        steps = 0

        # 🔑 USAR controller.max_steps (NO max_steps_per_episode)
        while not done and steps < controller.max_steps:
            episode_data["positions"].append(
                tuple(controller.env.agent_pos)
            )

            action = controller.agent.select_action(state, epsilon=0.0)
            next_state, reward, done, info = controller.env.step(action)

            episode_data["rewards"].append(reward)

            if info.get("success", False):
                episode_data["success"] = True

            state = next_state
            steps += 1

        trajectories.append(episode_data)

    # ----------------------------
    # Save
    # ----------------------------

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)

    print(f"[OK] Trajectories saved to: {output_path}")

    for t in trajectories:
        print(
            f"Ep {t['episode']} | Steps: {len(t['positions'])} | Success: {t['success']}"
        )


if __name__ == "__main__":
    main()
