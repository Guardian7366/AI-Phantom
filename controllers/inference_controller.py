import os
import numpy as np
from typing import Dict, Any, List, Optional


DEFAULT_BEST_MODEL_PATH = os.path.join(
    "results", "best_model", "best_model.pth"
)


class InferenceController:
    """
    Controlador de inferencia pura.
    Ejecuta episodios deterministas sin aprendizaje ni exploración.
    """

    def __init__(
        self,
        env,
        agent,
        model_path: Optional[str] = None,
        num_episodes: int = 10,
        max_steps_per_episode: int = 500,
        render: bool = False,
        render_delay: float = 0.0,
    ):
        self.env = env
        self.agent = agent

        # Resolución automática del modelo
        self.model_path = model_path or DEFAULT_BEST_MODEL_PATH
        self._validate_model_path()

        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode

        self.render = render
        self.render_delay = render_delay

        # Métricas descriptivas
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.successes: List[int] = []

    # ----------------------------
    # Validaciones
    # ----------------------------

    def _validate_model_path(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modelo de inferencia no encontrado: {self.model_path}"
            )

    # ----------------------------
    # Ejecución
    # ----------------------------

    def run(self) -> Dict[str, Any]:
        """
        Ejecuta episodios completos en modo inferencia pura.
        """
        # Configurar agente
        self.agent.set_mode(training=False)
        self.agent.load(self.model_path)

        for episode in range(1, self.num_episodes + 1):
            state = self.env.reset()

            episode_reward = 0.0
            success = False

            for step in range(self.max_steps):
                action = self.agent.select_action(
                    state,
                    epsilon=0.0  # inferencia 100% pura
                )

                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward
                state = next_state

                if self.render:
                    self.env.render()
                    if self.render_delay > 0:
                        import time
                        time.sleep(self.render_delay)

                if done:
                    success = info.get("success", False)
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            self.successes.append(1 if success else 0)

        return self._build_inference_summary()

    # ----------------------------
    # Métricas
    # ----------------------------

    def _build_inference_summary(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "episodes": self.num_episodes,
            "mean_reward": float(np.mean(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "success_rate": float(np.mean(self.successes)),
            "rewards": self.episode_rewards,
            "lengths": self.episode_lengths,
        }
