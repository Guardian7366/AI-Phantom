import numpy as np


class RandomAgent:
    def __init__(self, num_actions: int = 4, seed: int = 0):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic: bool = False) -> int:
        return int(self.rng.integers(0, self.num_actions))
