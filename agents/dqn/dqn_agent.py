import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.replay_buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

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
        device: Optional[str] = None,
    ):
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

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.training = True
        self.update_step = 0

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

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

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

        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        self.update_step += 1
        if self.update_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
