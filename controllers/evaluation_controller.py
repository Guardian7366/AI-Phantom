import numpy as np
from typing import Dict, Any, Callable, List
import random
import torch


class EvaluationController:

    """
    Evaluación robusta y reproducible.
    - Política determinista (epsilon=0)
    - Curriculum forzado explícitamente
    - Seed controlado por episodio
    - Compatible con randomize_grid
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_factory: Callable[[], Any],
        config: Dict[str, Any],
    ):
        self.env_factory = env_factory
        self.agent_factory = agent_factory
        self.config = config

        eval_cfg = config.get("evaluation", {})

        self.num_episodes: int = eval_cfg.get("num_episodes", 100)
        self.seed: int | None = eval_cfg.get("seed", 1234)

    # -------------------------------------------------

    def _set_global_seed(self, seed: int):

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------

    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        forced_curriculum_level: int | None = None,
    ) -> Dict[str, Any]:

        # Crear agente
        agent = self.agent_factory()
        agent.set_mode(training=False)
        agent.load(checkpoint_path)

        episode_rewards = []
        episode_lengths = []
        successes = []

        # -------------------------------------------------
        # Loop evaluación reproducible
        # -------------------------------------------------
        for episode_idx in range(self.num_episodes):

            # Seed distinto pero determinista por episodio
            if self.seed is not None:
                self._set_global_seed(self.seed + episode_idx)

            env = self.env_factory()

            # Forzar curriculum
            if forced_curriculum_level is not None:
                if hasattr(env, "set_curriculum_level"):
                    env.set_curriculum_level(forced_curriculum_level)

            max_steps = env.max_steps

            state = env.reset()

            episode_reward = 0.0
            success = False

            for step in range(max_steps):

                action = agent.select_action(state, epsilon=0.0)

                next_state, reward, done, info = env.step(action)

                episode_reward += reward
                state = next_state

                if done:
                    success = info.get("success", False)
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            successes.append(1 if success else 0)

        return self._build_summary(
            episode_rewards,
            episode_lengths,
            successes,
            forced_curriculum_level,
        )

    # -------------------------------------------------

    def _build_summary(
        self,
        rewards: List[float],
        lengths: List[int],
        successes: List[int],
        curriculum_level: int | None,
    ) -> Dict[str, Any]:

        rewards = np.array(rewards)
        lengths = np.array(lengths)
        successes = np.array(successes)

        return {
            "episodes": int(len(rewards)),
            "curriculum_level": curriculum_level,

            # Métricas centrales
            "success_rate": float(np.mean(successes)),
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),

            # Dispersión
            "reward_std": float(np.std(rewards)),
            "length_std": float(np.std(lengths)),

            # Percentiles
            "reward_p25": float(np.percentile(rewards, 25)),
            "reward_p50": float(np.percentile(rewards, 50)),
            "reward_p75": float(np.percentile(rewards, 75)),

            "length_p25": float(np.percentile(lengths, 25)),
            "length_p50": float(np.percentile(lengths, 50)),
            "length_p75": float(np.percentile(lengths, 75)),
        }
