# scripts/render_maze_ascii.py

import time
import yaml
from pathlib import Path

from environments.maze.maze_env import MazeEnvironment
from agents.dqn.dqn_agent import DQNAgent
from agents.dqn.replay_buffer import PrioritizedReplayBuffer


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    cfg_path = ROOT / "configs" / "maze_inference.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def render_ascii(env: MazeEnvironment):
    grid = env.grid
    agent_pos = tuple(env.agent_pos)
    goal_pos = env.goal

    rows, cols = grid.shape
    lines = []

    for r in range(rows):
        line = ""
        for c in range(cols):
            if (r, c) == agent_pos:
                line += "A"
            elif (r, c) == goal_pos:
                line += "G"
            elif grid[r, c] == 1:
                line += "#"
            else:
                line += "."
        lines.append(line)

    print("\n".join(lines))


def main():
    print("=== Render ASCII Maze ===")

    config = load_config()

    # ----------------------------
    # Environment
    # ----------------------------
    env = MazeEnvironment(config["environment"])
    state_dim = env.observation_space
    action_dim = env.action_space

    # ----------------------------
    # Agent
    # ----------------------------
    replay_buffer = PrioritizedReplayBuffer(capacity=1)  # dummy buffer

    agent_cfg = dict(config.get("agent", {}))

    # 🔥 limpiar metadata / parámetros inválidos
    agent_cfg.pop("type", None)
    agent_cfg.pop("epsilon", None)  # 👈 ESTE ERA EL BUG

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        **agent_cfg,
    )

    agent.set_mode(training=False)

    ckpt_path = ROOT / "results" / "best_model" / "best_model.pth"
    agent.load(str(ckpt_path))

    # ----------------------------
    # Run episode
    # ----------------------------
    state = env.reset()
    done = False
    step = 0
    max_steps = config.get("max_steps", 100)

    while not done and step < max_steps:
        print(f"\n--- Step {step} ---")
        render_ascii(env)

        action = agent.select_action(state, epsilon=0.0)
        state, reward, done, info = env.step(action)

        step += 1
        time.sleep(0.3)

    print("\n=== Episodio terminado ===")
    render_ascii(env)
    print(f"Steps: {step}")
    print(f"Success: {info.get('success', False)}")


if __name__ == "__main__":
    main()
