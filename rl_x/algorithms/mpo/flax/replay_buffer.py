import numpy as np


class ReplayBuffer():
    def __init__(self, capacity, nr_envs, trace_length, os_shape, as_shape, rng):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = capacity // nr_envs
        self.trace_length = trace_length
        self.nr_envs = nr_envs
        self.rng = rng
        self.states = np.zeros((self.capacity, nr_envs, trace_length) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs, trace_length) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs, trace_length), dtype=np.float32)
        self.terminations = np.zeros((self.capacity, nr_envs, trace_length), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity, nr_envs, trace_length), dtype=np.float32)
        self.pos = 0
        self.size = 0
    

    def add(self, states, actions, rewards, terminations, log_probs):
        self.states[self.pos] = states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.terminations[self.pos] = terminations
        self.log_probs[self.pos] = log_probs
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, nr_samples):
        idx1 = self.rng.integers(self.size, size=nr_samples)
        idx2 = self.rng.integers(self.nr_envs, size=nr_samples)
        states = self.states[idx1, idx2].reshape((nr_samples, self.trace_length) + self.os_shape)
        actions = self.actions[idx1, idx2].reshape((nr_samples, self.trace_length) + self.as_shape)
        rewards = self.rewards[idx1, idx2].reshape((nr_samples, self.trace_length))
        terminations = self.terminations[idx1, idx2].reshape((nr_samples, self.trace_length))
        log_probs = self.log_probs[idx1, idx2].reshape((nr_samples, self.trace_length))
        return states, actions, rewards, terminations, log_probs
