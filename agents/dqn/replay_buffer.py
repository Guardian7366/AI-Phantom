import numpy as np


# ============================================================
# SUM TREE
# ============================================================

class SumTree:
    """
    Binary tree where parent = sum(children).
    Used for efficient proportional sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.full(capacity, None, dtype=object)
        self.size = 0
        self.write = 0

    # ---------------------------------------------------------

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # ---------------------------------------------------------

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # ---------------------------------------------------------

    def total(self) -> float:
        return self.tree[0]

    # ---------------------------------------------------------

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ---------------------------------------------------------

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    # ---------------------------------------------------------

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ============================================================
# PRIORITIZED REPLAY BUFFER (PROPORTIONAL)
# ============================================================

class PrioritizedReplayBuffer:

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        epsilon: float = 1e-5
    ):
        """
        alpha: how much prioritization is used (0 = uniform)
        epsilon: small value to avoid zero priority
        """
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon

        self.tree = SumTree(capacity)

        self.max_priority = 1.0

    # ---------------------------------------------------------

    def push(self, state, action, reward, next_state, done):

        data = (state, action, reward, next_state, done)

        # New transitions inserted with max priority
        priority = self.max_priority
        self.tree.add(priority, data)

    # ---------------------------------------------------------

    def sample(self, batch_size: int, beta: float = 0.4):

        if self.tree.size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        batch = []
        indices = []
        priorities = []

        total_priority = self.tree.total()

        if total_priority == 0:
            raise ValueError("SumTree total priority is zero")

        segment = total_priority / batch_size

        for i in range(batch_size):
            while True:
                a = segment * i
                b = segment * (i + 1)
                s = np.random.uniform(a, b)

                idx, priority, data = self.tree.get(s)

                # 🔥 VALIDACIÓN CRÍTICA
                data_idx = idx - self.tree.capacity + 1

                if data is None:
                    continue

                if data_idx < 0 or data_idx >= self.tree.size:
                    continue

                break

            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        states, actions, rewards, next_states, dones = zip(*batch)

        sampling_probabilities = np.array(priorities, dtype=np.float32) / total_priority

        weights = (self.tree.size * sampling_probabilities) ** (-beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )


    # ---------------------------------------------------------

    def update_priorities(self, indices, td_errors):

        td_errors = np.abs(td_errors) + self.epsilon

        for idx, error in zip(indices, td_errors):
            priority = (error ** self.alpha)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    # ---------------------------------------------------------

    def __len__(self):
        return self.tree.size
