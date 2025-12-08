import numpy as np

class ReplayMemory:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        # ローカル RNG
        self.rng = np.random.default_rng(seed)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (
            np.copy(state),
            action,
            reward,
            np.copy(next_state),
            done
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = self.rng.integers(len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in idx]

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done.astype(np.float32)

    def __len__(self):
        return len(self.buffer)
