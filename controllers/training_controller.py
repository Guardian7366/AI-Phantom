import os
import json
import numpy as np
from collections import deque
from typing import Dict, Any
from datetime import datetime

from controllers.evaluation_controller import EvaluationController


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
        results_dir: str = "results/runs",
        history_dir: str = "results/histories",
        early_stopping_patience: int = 50,
        success_threshold: float = 0.95,
        success_window: int = 100,
        experiment_id: str | None = None,
    ):
        self.env = env
        self.agent = agent

        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.history_dir = history_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        self.early_stopping_patience = early_stopping_patience
        self.success_threshold = success_threshold
        self.success_window = success_window

        self.experiment_id = (
            experiment_id
            if experiment_id is not None
            else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        )

        # Métricas principales
        self.episode_rewards = []
        self.episode_lengths = []
        self.full_success_history = []
        self.loss_history = []
        self.epsilon_history = []
        self.rolling_success_rate = []

        self.success_history = deque(maxlen=success_window)

        self.best_success_rate = 0.0
        self.no_improve_counter = 0

    # -------------------------------------------------

    def _compute_epsilon(self, episode: int) -> float:
        if episode >= self.epsilon_decay_episodes:
            return self.epsilon_end

        decay_ratio = episode / self.epsilon_decay_episodes
        return self.epsilon_start - decay_ratio * (
            self.epsilon_start - self.epsilon_end
        )

    # -------------------------------------------------

    def train(self) -> Dict[str, Any]:
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

            # Registrar métricas
            self.episode_rewards.append(float(episode_reward))
            self.episode_lengths.append(step + 1)

            success_int = 1 if success else 0
            self.success_history.append(success_int)
            self.full_success_history.append(success_int)

            rolling_rate = float(np.mean(self.success_history))
            self.rolling_success_rate.append(rolling_rate)

            self.loss_history.append(
                float(np.mean(episode_loss)) if episode_loss else 0.0
            )

            self.epsilon_history.append(float(epsilon))

            if self._check_and_handle_progress(episode):
                break

        # Guardar modelos
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        self.agent.save(last_path)

        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_path):
            self.agent.save(best_path)

        evaluation_results = self._run_evaluation(best_path)

        training_summary = self._build_training_summary()

        experiment_results = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "training": training_summary,
            "evaluation": evaluation_results,
        }

        # Guardar JSON principal
        results_path = os.path.join(
            self.results_dir, f"{self.experiment_id}.json"
        )

        with open(results_path, "w") as f:
            json.dump(experiment_results, f, indent=2)

        # Guardar historial separado (sandbox-ready)
        self._save_history_file()

        return experiment_results

    # -------------------------------------------------

    def _save_history_file(self):

        history_data = {
            "experiment_id": self.experiment_id,
            "episodes": list(range(1, len(self.episode_rewards) + 1)),
            "reward": self.episode_rewards,
            "length": self.episode_lengths,
            "success": self.full_success_history,
            "loss": self.loss_history,
            "epsilon": self.epsilon_history,
            "rolling_success_rate": self.rolling_success_rate,
        }

        history_path = os.path.join(
            self.history_dir,
            f"{self.experiment_id}_history.json"
        )

        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)

    # -------------------------------------------------

    def _check_and_handle_progress(self, episode: int) -> bool:
        if len(self.success_history) < self.success_window:
            return False

        success_rate = float(np.mean(self.success_history))

        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.no_improve_counter = 0

            checkpoint_path = os.path.join(
                self.checkpoint_dir, "best_model.pth"
            )
            self.agent.save(checkpoint_path)
        else:
            self.no_improve_counter += 1

        if success_rate >= self.success_threshold:
            print(
                f"[Early Stop] Success rate alcanzado: "
                f"{success_rate:.2f} en episodio {episode}"
            )
            return True

        if self.no_improve_counter >= self.early_stopping_patience:
            print(
                f"[Early Stop] Sin mejora durante "
                f"{self.early_stopping_patience} ventanas"
            )
            return True

        return False

    # -------------------------------------------------

    def _run_evaluation(self, checkpoint_path: str) -> Dict[str, Any]:
        evaluator = EvaluationController(
            env_factory=self.env.factory,
            agent_factory=self.agent.factory,
            config=self.env.config,
        )

        return evaluator.evaluate_checkpoint(checkpoint_path)

    # -------------------------------------------------

    def _build_training_summary(self) -> Dict[str, Any]:
        return {
            "episodes": len(self.episode_rewards),
            "mean_reward": float(np.mean(self.episode_rewards)),
            "mean_length": float(np.mean(self.episode_lengths)),
            "best_success_rate": float(self.best_success_rate),
            "final_epsilon": self.epsilon_history[-1]
            if self.epsilon_history
            else 0.0,
            "reward_history": self.episode_rewards,
            "length_history": self.episode_lengths,
            "success_history": self.full_success_history,
            "loss_history": self.loss_history,
        }
