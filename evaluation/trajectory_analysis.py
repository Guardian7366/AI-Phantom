import json
from typing import List, Dict


def collect_trajectories(env, agent, episodes: int = 5) -> List[Dict]:
    """
    Ejecuta episodios y guarda las trayectorias completas.
    """
    trajectories = []

    agent.set_mode(training=False)

    for ep in range(episodes):
        state = env.reset()
        done = False

        episode_traj = {
            "episode": ep,
            "positions": [],
            "success": False,
        }

        while not done:
            episode_traj["positions"].append(tuple(env.agent_pos))

            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, info = env.step(action)

            if info.get("success", False):
                episode_traj["success"] = True

        trajectories.append(episode_traj)

    return trajectories

def save_trajectories(trajectories: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)
