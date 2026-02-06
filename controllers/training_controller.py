import os
import numpy as np
from collections import deque
from typing import Dict, Any

class TrainingController:
    """
    Controlador de entrenamiento.
    Coordina Environment y DQNAgent sin contener lógica de aprendizaje.
    """

    def __init__(
        self,
        env,
        agent,
        num_episodes: int,
        max_steps_per_episode: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 500,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 50,
        success_threshold: float = 0.95,
        success_window: int = 100,
    ):
        self.env = env
        self.agent = agent

        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.early_stopping_patience = early_stopping_patience
        self.success_threshold = success_threshold
        self.success_window = success_window

        # Métricas
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_history = deque(maxlen=success_window)
        self.loss_history = []

        self.best_success_rate = 0.0
        self.no_improve_counter = 0

    def _compute_epsilon(self, episode: int) -> float:
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end

        decay_ratio = episode / self.epsilon_decay_episodes
        return self.epsilon_start - decay_ratio * (
            self.epsilon_start - self.epsilon_end
        )

    def train(self) -> Dict[str, Any]:
        """
        Ejecuta el entrenamiento completo.
        """
        self.agent.set_mode(training=True)

        for episode in range(1, self.num_episodes + 1):
            state = self.env.reset()

            epsilon = self._compute_epsilon(episode)

            episode_reward = 0.0
            episode_loss = []
            success = False

            for step in range(self.max_steps):
                action = self.agent.select_action(state, epsilon)

                next_state, reward, done, info = self.env.step(action)

                self.agent.observe(state, action, reward, next_state, done)

                loss = self.agent.update()
                if loss is not None:
                    episode_loss.append(loss)

                episode_reward += reward
                state = next_state

                if done:
                    success = info.get("success", False)
                    break

            # Registrar métricas por episodio
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step + 1)
            self.success_history.append(1 if success else 0)

            if episode_loss:
                self.loss_history.append(np.mean(episode_loss))

            # Evaluar progreso
            if self._check_and_handle_progress(episode):
                break

        return self._build_training_summary()

    def _check_and_handle_progress(self, episode: int) -> bool:
        """
        Evalúa métricas agregadas y aplica early stopping.
        Devuelve True si debe detener el entrenamiento.
        """
        if len(self.success_history) < self.success_window:
            return False

        success_rate = np.mean(self.success_history)

        # Guardar mejor modelo
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.no_improve_counter = 0

            checkpoint_path = os.path.join(
                self.checkpoint_dir, "best_model.pth"
            )
            self.agent.save(checkpoint_path)
        else:
            self.no_improve_counter += 1

        # Early stopping por convergencia real
        if success_rate >= self.success_threshold:
            print(
                f"[Early Stop] Success rate alcanzado: {success_rate:.2f} "
                f"en episodio {episode}"
            )
            return True

        # Early stopping por estancamiento
        if self.no_improve_counter >= self.early_stopping_patience:
            print(
                f"[Early Stop] Sin mejora durante "
                f"{self.early_stopping_patience} ventanas"
            )
            return True

        return False

    def _build_training_summary(self) -> Dict[str, Any]:
        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "best_success_rate": float(self.best_success_rate),
            "final_epsilon": self._compute_epsilon(len(self.episode_rewards)),
        }
