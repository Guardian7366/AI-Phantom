import random
from typing import Optional, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.replay_buffer import PrioritizedReplayBuffer


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
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )


        conv_out_size = 128 * h * w

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.LayerNorm(256),
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
        replay_buffer: PrioritizedReplayBuffer,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        min_replay_size: int = 1000,
        tau: float = 0.005,
        update_frequency: int = 4,
        device: Optional[str] = None
    ):

        self._init_params = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        tau=tau,
        update_frequency=update_frequency,
        device=device,
    )
        self.beta_start = 0.4
        self.beta_frames = 200000
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.tau = tau
        self.update_frequency = update_frequency
        self.step_counter = 0


        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------
        # AMP GradScaler initialization
        # ------------------------------
        if self.device == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        self.policy_net = CNN_Dueling_DQN(state_dim, action_dim).to(self.device)
        self.target_net = CNN_Dueling_DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self.training = True

        self.factory: Callable[[], "DQNAgent"] = self._build_factory

    def _build_factory(self):
        replay_buffer = PrioritizedReplayBuffer(capacity=1)
        agent = DQNAgent(replay_buffer=replay_buffer, **self._init_params)
        agent.set_mode(False)
        return agent

    def set_mode(self, training: bool):
        self.training = training
        self.policy_net.train(training)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:

        if self.training and random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.step_counter += 1


    def update(self):
        if self.step_counter % self.update_frequency != 0:
            return None

        if not self.training:
            return None

        if len(self.replay_buffer) < max(self.batch_size, self.min_replay_size):
            return None

        beta = self._beta_by_frame()
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size, beta)


        weights = torch.from_numpy(weights).to(self.device).unsqueeze(1)

        states = torch.from_numpy(states).to(self.device, non_blocking=True)
        actions = torch.from_numpy(actions).long().to(self.device, non_blocking=True).unsqueeze(1)
        rewards = torch.from_numpy(rewards).to(self.device, non_blocking=True).unsqueeze(1)
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones).to(self.device, non_blocking=True).unsqueeze(1)


        # ------------------------------
        # Mixed Precision Training
        # ------------------------------
        use_amp = self.device == "cuda"

        if use_amp:
            with torch.amp.autocast("cuda"):
                q_values = self.policy_net(states).gather(1, actions)

                with torch.no_grad():
                    next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                    next_q = self.target_net(next_states).gather(1, next_actions)
                    target_q = rewards + self.gamma * next_q * (1 - dones)
                    target_q = target_q.detach()

                td_error = q_values - target_q
                loss = (weights * self.loss_fn(q_values, target_q)).mean()


            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            td_errors = td_error.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)


        else:
            q_values = self.policy_net(states).gather(1, actions)

            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards + self.gamma * next_q * (1 - dones)

            td_error = q_values - target_q
            loss = (weights * self.loss_fn(q_values, target_q)).mean()


            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            td_errors = td_error.detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)



        # ------------------------------
        # Soft target update
        # ------------------------------
        for target_param, policy_param in zip(
            self.target_net.parameters(),
            self.policy_net.parameters()
        ):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * policy_param.data)

        return float(loss.item())

   

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _beta_by_frame(self):
        return min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) *
            (self.step_counter / self.beta_frames)
        )

