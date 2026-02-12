import numpy as np
from typing import Dict, Any, Callable, List


class EvaluationController:
    """
    Controlador de evaluación cuantitativa.
    Ejecuta episodios deterministas usando factories para reproducibilidad.
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
        self.max_steps: int = eval_cfg.get("max_steps_per_episode", 500)
        self.seed: int | None = eval_cfg.get("seed", None)

    # -------------------------------------------------
    # API pública
    # -------------------------------------------------

    def evaluate_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Ejecuta evaluación del modelo guardado en checkpoint_path.
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        env = self.env_factory()
        agent = self.agent_factory()

        agent.set_mode(training=False)
        agent.load(checkpoint_path)

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        successes: List[int] = []

        for _ in range(self.num_episodes):
            state = env.reset()

            episode_reward = 0.0
            success = False

            for step in range(self.max_steps):
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

        return self._build_evaluation_summary(
            episode_rewards,
            episode_lengths,
            successes,
        )

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _build_evaluation_summary(
        self,
        rewards: List[float],
        lengths: List[int],
        successes: List[int],
    ) -> Dict[str, Any]:

        rewards = np.array(rewards)
        lengths = np.array(lengths)
        successes = np.array(successes)

        return {
            "episodes": int(self.num_episodes),

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
