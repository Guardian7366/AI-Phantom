import numpy as np
from typing import Dict, Any, List

class EvaluationController:
    """
    Controlador de evaluación cuantitativa.
    Ejecuta episodios deterministas para medir desempeño reproducible.
    """

    def __init__(
        self,
        env,
        agent,
        model_path: str,
        num_episodes: int = 100,
        max_steps_per_episode: int = 500,
        seed: int | None = None,
    ):
        self.env = env
        self.agent = agent
        self.model_path = model_path

        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.seed = seed

        # Métricas crudas por episodio
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.successes: List[int] = []

    def run(self) -> Dict[str, Any]:
        """
        Ejecuta episodios de evaluación sin aprendizaje ni exploración.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.agent.set_mode(training=False)
        self.agent.load(self.model_path)

        for _ in range(self.num_episodes):
            state = self.env.reset()

            episode_reward = 0.0
            success = False

            for step in range(self.max_steps):
                action = self.agent.select_action(state, epsilon=0.0)

                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                state = next_state

                if done:
                    success = info.get("success", False)
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            self.successes.append(1 if success else 0)

        return self._build_evaluation_summary()

    def _build_evaluation_summary(self) -> Dict[str, Any]:
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)
        successes = np.array(self.successes)

        return {
            "episodes": self.num_episodes,

            # Métricas centrales
            "success_rate": float(np.mean(successes)),
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),

            # Dispersión
            "reward_std": float(np.std(rewards)),
            "length_std": float(np.std(lengths)),

            # Percentiles (muy importantes)
            "reward_p25": float(np.percentile(rewards, 25)),
            "reward_p50": float(np.percentile(rewards, 50)),
            "reward_p75": float(np.percentile(rewards, 75)),

            "length_p25": float(np.percentile(lengths, 25)),
            "length_p50": float(np.percentile(lengths, 50)),
            "length_p75": float(np.percentile(lengths, 75)),
        }
