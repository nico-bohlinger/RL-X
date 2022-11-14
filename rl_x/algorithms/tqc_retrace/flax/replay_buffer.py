import numpy as np


class ReplayBuffer():
    def __init__(self, capacity, trace_length, os_shape, as_shape):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = capacity
        self.trace_length = trace_length
        self.states = np.zeros((self.capacity, trace_length) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, trace_length) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, trace_length) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, trace_length), dtype=np.float32)
        self.dones = np.zeros((self.capacity, trace_length), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity, trace_length), dtype=np.float32)
        self.pos = 0
        self.size = 0
    

    def add(self, states, next_states, actions, rewards, dones, log_probs):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.log_probs[self.pos] = log_probs
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, nr_samples, nr_batches):
        idx1 = np.random.randint(self.size, size=nr_samples * nr_batches)
        states = self.states[idx1].reshape((nr_batches, nr_samples, self.trace_length) + self.os_shape)
        next_states = self.next_states[idx1].reshape((nr_batches, nr_samples, self.trace_length) + self.os_shape)
        actions = self.actions[idx1].reshape((nr_batches, nr_samples, self.trace_length) + self.as_shape)
        rewards = self.rewards[idx1].reshape((nr_batches, nr_samples, self.trace_length))
        dones = self.dones[idx1].reshape((nr_batches, nr_samples, self.trace_length))
        log_probs = self.log_probs[idx1].reshape((nr_batches, nr_samples, self.trace_length))
        return states, next_states, actions, rewards, dones, log_probs
