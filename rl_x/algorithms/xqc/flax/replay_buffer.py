import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, nr_envs, os_shape, as_shape, rng):
        self.capacity = capacity // nr_envs
        self.nr_envs = nr_envs
        self.rng = rng
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.terminations = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0


    def add(self, state, next_state, action, reward, terminated):
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.terminations[self.pos] = terminated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size):
        idx_t = self.rng.integers(0, self.size, size=batch_size)
        idx_e = self.rng.integers(0, self.nr_envs, size=batch_size)
        return (
            self.states[idx_t, idx_e],
            self.next_states[idx_t, idx_e],
            self.actions[idx_t, idx_e],
            self.rewards[idx_t, idx_e],
            self.terminations[idx_t, idx_e],
        )
