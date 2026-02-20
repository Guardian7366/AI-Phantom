from __future__ import annotations

from typing import Dict, Any


class InferenceController:
    """
    Controller mínimo y limpio:
    - Ejecuta un episodio determinista para integrarlo con sandbox.
    """
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run_episode(self, curriculum_level: int = 2, seed: int = 123) -> Dict[str, Any]:
        obs, info = self.env.reset(curriculum_level=curriculum_level, seed=seed)
        done = False
        trunc = False

        episode = {
            "grid": self.env.grid.tolist(),
            "start": list(self.env.agent_pos),
            "goal": list(self.env.goal_pos),
            "steps": [],
            "meta": {"curriculum_level": int(curriculum_level), "seed": int(seed)},
        }

        while not (done or trunc):
            a = self.agent.act(obs, deterministic=True)
            prev = self.env.agent_pos
            obs, r, done, trunc, info = self.env.step(a)
            episode["steps"].append({
                "action": int(a),
                "reward": float(r),
                "from": list(prev),
                "to": list(self.env.agent_pos),
                "done": bool(done),
                "trunc": bool(trunc),
            })

        episode["meta"]["success"] = bool(done)
        episode["meta"]["num_steps"] = len(episode["steps"])
        return episode
