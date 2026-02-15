# scripts/render_maze_trajectory.py

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


def render_with_trajectory(env: MazeEnvironment, trajectory: list[tuple[int, int]]):
    """
    Renderiza el laberinto con la trayectoria superpuesta.
    """
    grid = env.grid
    start = tuple(env.start)
    goal = tuple(env.goal)

    rows, cols = grid.shape
    canvas = []

    trajectory_set = set(trajectory[1:-1])  # sin inicio ni goal

    for r in range(rows):
        line = ""
        for c in range(cols):
            pos = (r, c)

            if pos == start:
                line += "S"
            elif pos == goal:
                line += "G"
            elif pos in trajectory_set:
                line += "*"
            elif grid[r, c] == 1:
                line += "#"
            else:
                line += "."
        canvas.append(line)

    print("\n".join(canvas))


def main():
    print("=== Render Maze Trajectory ===")

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
    replay_buffer = PrioritizedReplayBuffer(capacity=1)  # dummy

    agent_cfg = dict(config.get("agent", {}))
    agent_cfg.pop("type", None)      # metadata
    agent_cfg.pop("epsilon", None)   # no pertenece al init

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
    # Run episode & record path
    # ----------------------------
    state = env.reset()
    trajectory = [tuple(env.agent_pos)]

    done = False
    step = 0
    max_steps = config["environment"].get("max_steps", 100)

    while not done and step < max_steps:
        action = agent.select_action(state, epsilon=0.0)
        state, reward, done, info = env.step(action)

        trajectory.append(tuple(env.agent_pos))
        step += 1

    print("\n--- Trayectoria final ---")
    render_with_trajectory(env, trajectory)

    print("\nSteps:", step)
    print("Success:", info.get("success", False))


if __name__ == "__main__":
    main()
