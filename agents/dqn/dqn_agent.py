import random
from typing import Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.replay_buffer import PrioritizedReplayBuffer


# ============================================================
# NETWORK: CNN + DUELING (Optimizada para Maze 8x8)
# ============================================================

class CNN_Dueling_DQN(nn.Module):

    def __init__(self, state_shape: Tuple[int, int, int], action_dim: int):
        super().__init__()

        c, h, w = state_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()

        conv_out = 32 * h * w

        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out, 128),
            nn.ReLU(inplace=True),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.shared_fc(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + advantage - advantage.mean(dim=1, keepdim=True)


# ============================================================
# AGENT
# ============================================================

class DQNAgent:

    def __init__(
        self,
        state_dim: Tuple[int, int, int],
        action_dim: int,
        replay_buffer: PrioritizedReplayBuffer,
        gamma: float = 0.99,
        lr: float = 3e-4,
        batch_size: int = 64,
        min_replay_size: int = 2000,
        tau: float = 0.005,
        update_frequency: int = 4,
        n_step: int = 3,
        device: Optional[str] = None
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.tau = tau
        self.update_frequency = update_frequency
        self.n_step = n_step

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = CNN_Dueling_DQN(state_dim, action_dim).to(self.device)
        self.target_net = CNN_Dueling_DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # AMP
        self.use_amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Training state
        self.training = True
        self.step_counter = 0
        self.n_step_buffer = []

        # Prioritized replay beta schedule
        self.beta_start = 0.4
        self.beta_frames = 300_000

        self.factory: Callable[[], "DQNAgent"] = self._build_factory

    # ============================================================
    # MODE CONTROL
    # ============================================================

    def set_mode(self, training: bool):
        self.training = training
        self.policy_net.train(training)

    def _build_factory(self):
        buffer = PrioritizedReplayBuffer(capacity=1)
        agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            replay_buffer=buffer,
            gamma=self.gamma,
            lr=self.optimizer.param_groups[0]['lr'],
            batch_size=self.batch_size,
            min_replay_size=self.min_replay_size,
            tau=self.tau,
            update_frequency=self.update_frequency,
            n_step=self.n_step,
            device=self.device
        )
        agent.set_mode(False)
        return agent

    # ============================================================
    # ACTION SELECTION
    # ============================================================

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:

        if self.training and random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    # ============================================================
    # EXPERIENCE COLLECTION (N-STEP)
    # ============================================================

    def observe(self, state, action, reward, next_state, done):

        if not self.training:
            return

        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        cumulative_reward = 0.0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            cumulative_reward += (self.gamma ** idx) * r
            if d:
                break

        first_state, first_action = self.n_step_buffer[0][:2]
        last_next_state, last_done = self.n_step_buffer[-1][3:5]

        self.replay_buffer.push(
            first_state,
            first_action,
            cumulative_reward,
            last_next_state,
            last_done
        )

        self.n_step_buffer.pop(0)
        self.step_counter += 1

        if done:
            self.n_step_buffer.clear()

    # ============================================================
    # UPDATE STEP (Double DQN + PER)
    # ============================================================

    def update(self):

        if not self.training:
            return None

        if self.step_counter % self.update_frequency != 0:
            return None

        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None

        beta = min(
            1.0,
            self.beta_start +
            (1.0 - self.beta_start) *
            (self.step_counter / self.beta_frames)
        )

        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size, beta)

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(rewards).to(self.device).unsqueeze(1)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device).unsqueeze(1)
        weights = torch.from_numpy(weights).to(self.device).unsqueeze(1)

        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                current_q = self.policy_net(states).gather(1, actions)
                loss_elements = self.loss_fn(current_q, target_q)
                loss = (weights * loss_elements).mean()

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            current_q = self.policy_net(states).gather(1, actions)
            loss_elements = self.loss_fn(current_q, target_q)
            loss = (weights * loss_elements).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        # Update priorities
        td_errors = (current_q - target_q).detach().abs().cpu().numpy().flatten()
        td_errors = np.clip(td_errors, 1e-6, 10.0)
        self.replay_buffer.update_priorities(indices, td_errors)

        self._soft_update()

        return float(loss.item())

    # ============================================================
    # TARGET UPDATE
    # ============================================================

    def _soft_update(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                policy_param.data * self.tau
            )

    # ============================================================
    # SAVE / LOAD
    # ============================================================

    def save(self, path: str):
        torch.save({
            "model_state_dict": self.policy_net.state_dict(),
        }, path)

    def load(self, path: str):  
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())

