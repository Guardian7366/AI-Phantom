import random
from typing import Optional, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.replay_buffer import ReplayBuffer


# =====================================================
# CNN DUELING DQN
# =====================================================

class CNN_Dueling_DQN(nn.Module):

    def __init__(self, state_shape: Tuple[int, int, int], action_dim: int):
        super().__init__()

        c, h, w = state_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * h * w

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# =====================================================
# AGENT
# =====================================================

class DQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim: int,
        replay_buffer: ReplayBuffer,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        min_replay_size: int = 1000,
        device: Optional[str] = None
    ):

        self._init_params = dict(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            device=device,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = CNN_Dueling_DQN(state_dim, action_dim).to(self.device)
        self.target_net = CNN_Dueling_DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.training = True
        self.tau = 0.005

        self.factory: Callable[[], "DQNAgent"] = self._build_factory

    def _build_factory(self):
        replay_buffer = ReplayBuffer(capacity=1)
        agent = DQNAgent(replay_buffer=replay_buffer, **self._init_params)
        agent.set_mode(False)
        return agent

    def set_mode(self, training: bool):
        self.training = training
        self.policy_net.train(training)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:

        if self.training and random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32)\
            .unsqueeze(0)\
            .to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):

        if not self.training:
            return None

        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data +
                (1.0 - self.tau) * target_param.data
            )

        return loss.item()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
