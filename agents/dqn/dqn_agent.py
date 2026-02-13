import random
from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.replay_buffer import ReplayBuffer


class DQN(nn.Module):
    """
    Dueling DQN Architecture
    Separa Value y Advantage streams.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        features = self.feature_layer(x)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values



class DQNAgent:
    """
    Agente DQN desacoplado del entorno y del controller.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer: ReplayBuffer,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        min_replay_size: int = 1000,
        device: Optional[str] = None
    ):
        # Guardar parámetros para factory
        self._init_params = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
        )
        self.tau = 0.005
        self.min_replay_size = min_replay_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            eps=1e-5
        )

        self.loss_fn = nn.SmoothL1Loss()

        self.training = True
        self.update_step = 0

        # Factory para evaluación (CRÍTICO)
        self.factory: Callable[[], "DQNAgent"] = self._build_factory

    # ---------------------
    # Factory
    # ---------------------

    def _build_factory(self) -> "DQNAgent":
        """
        Crea una nueva instancia del agente (sin replay buffer compartido).
        Usado exclusivamente para evaluación.
        """
        replay_buffer = ReplayBuffer(capacity=1)  # dummy buffer

        agent = DQNAgent(
            replay_buffer=replay_buffer,
            **self._init_params,
        )

        agent.set_mode(training=False)
        return agent

    # ---------------------
    # API
    # ---------------------

    def set_mode(self, training: bool) -> None:
        """
        Define si el agente está en modo entrenamiento o inferencia.
        """
        self.training = training
        self.policy_net.train(training)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Selecciona una acción.
        - En entrenamiento: epsilon-greedy
        - En inferencia: greedy puro
        """
        if self.training and random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Guarda la transición en el replay buffer.
        """
        if not self.training:
            return

        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """
        Ejecuta un paso de entrenamiento si hay suficientes datos.
        Devuelve la pérdida si se entrenó, None si no.
        """
        if not self.training:
            return None

        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            # Acción seleccionada por la policy net
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

            # Evaluación con target net
            next_q_values = self.target_net(next_states).gather(1, next_actions)

            target_q = rewards + self.gamma * next_q_values * (1 - dones)


        loss = self.loss_fn(q_values, target_q)
        q_reg = 1e-5 * q_values.pow(2).mean()
        loss = loss + q_reg

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)   
        self.optimizer.step()

        self.update_step += 1
        
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters(),
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

        return loss.item()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

