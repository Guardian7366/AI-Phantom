import os
import json
import numpy as np
from collections import deque
from typing import Dict, Any
from datetime import datetime

from controllers.evaluation_controller import EvaluationController
from environments.maze.maze_env import MazeEnvironment


class TrainingController:

    def __init__(
        self,
        env: MazeEnvironment,
        agent,
        num_episodes: int,
        config: Dict[str, Any],
        max_steps_per_episode: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 2000,  # 🔥 antes 500
        checkpoint_dir: str = "checkpoints",
        results_dir: str = "results/runs",
        history_dir: str = "results/histories",
        early_stopping_patience: int = 100,
        success_threshold: float = 0.98,
        success_window: int = 100,
        experiment_id: str | None = None,
    ):

        self.env = env
        self.agent = agent
        self.config = config
        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode

        self.env.max_steps = self.max_steps

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes

        self.experiment_id = (
            experiment_id
            if experiment_id is not None
            else datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        )

        self.checkpoint_dir = os.path.join(checkpoint_dir, self.experiment_id)
        self.results_dir = os.path.join(results_dir)
        self.history_dir = os.path.join(history_dir)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        self.early_stopping_patience = early_stopping_patience
        self.success_threshold = success_threshold
        self.success_window = success_window

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

        # Boost después de subir curriculum
        if hasattr(self, "_curriculum_boost_counter"):
            if self._curriculum_boost_counter < self._curriculum_boost_episodes:
                self._curriculum_boost_counter += 1
                return 0.4  # 🔥 exploración controlada

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

                # 🔥 No actualizar si buffer muy pequeño
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    loss = self.agent.update()
                    if loss is not None:
                        episode_loss.append(loss)

                episode_reward += reward
                state = next_state

                if done:
                    success = info.get("success", False)
                    break

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

        # Guardar último modelo
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        self.agent.save(last_path)

        evaluation_last = self._run_evaluation(last_path)

        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.agent.save(best_path)

        training_summary = self._build_training_summary()

        experiment_results = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "training": training_summary,
            "evaluation": evaluation_last,
        }

        results_path = os.path.join(
            self.results_dir, f"{self.experiment_id}.json"
        )

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=2)

        self._save_history_file()

        return experiment_results

    # -------------------------------------------------

    def _check_and_handle_progress(self, episode: int) -> bool:

        if len(self.success_history) < self.success_window:
            return False

        success_rate = float(np.mean(self.success_history))

        # 🔥 Curriculum más estable
        if hasattr(self.env, "curriculum_level"):

            if not hasattr(self, "_curriculum_cooldown"):
                self._curriculum_cooldown = False

            if (
                success_rate >= 0.92  # 🔥 antes 0.90
                and self.env.curriculum_level < 4
                and not self._curriculum_cooldown
            ):

                new_level = self.env.curriculum_level + 1
                self.env.set_curriculum_level(new_level)

                if hasattr(self.agent.replay_buffer, "priorities"):
                    self.agent.replay_buffer.priorities *= 0.7

                self._curriculum_boost_episodes = 300
                self._curriculum_boost_counter = 0

                self._curriculum_cooldown = True
                print(f"[Curriculum] Subiendo a nivel {new_level}")

            if success_rate < 0.70:
                self._curriculum_cooldown = False

        # Best model tracking
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1

        # Early stop
        if (
            self.success_threshold is not None
            and success_rate >= self.success_threshold
        ):
            print(
                f"[Early Stop] Success rate alcanzado: "
                f"{success_rate:.3f} en episodio {episode}"
            )
            return True

        if (
            self.early_stopping_patience is not None
            and self.no_improve_counter >= self.early_stopping_patience
        ):
            print(
                f"[Early Stop] Sin mejora durante "
                f"{self.early_stopping_patience} ventanas"
            )
            return True

        return False

    # -------------------------------------------------

    def _run_evaluation(self, checkpoint_path: str) -> Dict[str, Any]:

        def env_factory():
            new_env = MazeEnvironment(self.config["environment"])
            new_env.set_curriculum_level(
                getattr(self.env, "curriculum_level", 0)
            )
            return new_env

        evaluator = EvaluationController(
            env_factory=env_factory,
            agent_factory=self.agent.factory,
            config=self.config,
        )

        final_level = getattr(self.env, "curriculum_level", 0)

        return evaluator.evaluate_checkpoint(
            checkpoint_path,
            forced_curriculum_level=final_level,
        )

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
            "final_curriculum_level": getattr(self.env, "curriculum_level", 0),
        }

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

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)
