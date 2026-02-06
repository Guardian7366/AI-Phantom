import random
from collections import deque
from typing import Tuple, List
import numpy as np


class ReplayBuffer:
    """
    Replay Buffer clásico para DQN.
    Almacena transiciones y permite muestreo aleatorio.
    """

    def __init__(self, capacity: int):
        """
        capacity: número máximo de transiciones almacenadas
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Guarda una transición en el buffer.
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple:
        """
        Devuelve un batch aleatorio de transiciones.
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """
        Vacía el buffer completamente.
        """
        self.buffer.clear()
