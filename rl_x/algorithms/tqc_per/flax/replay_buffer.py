import numpy as np
import time

from rl_x.algorithms.tqc_per.flax.priority_tree import PriorityTree


class ReplayBuffer():
    def __init__(self, capacity, os_shape, as_shape, per_alpha, per_beta, per_epsilon, per_start_priority):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_epsilon = per_epsilon
        self.capacity = capacity
        self.states = np.zeros((self.capacity, ) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, ) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, ) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, ), dtype=np.float32)
        self.dones = np.zeros((self.capacity, ), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.highest_priority = per_start_priority
        self.priority_tree = PriorityTree(self.capacity)
    

    def add(self, states, next_states, actions, rewards, dones):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.priority_tree.update(self.pos, self.highest_priority ** self.per_alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, nr_samples, nr_batches):
        idx, priority = self.priority_tree.sample(nr_samples * nr_batches)
        is_weights = (self.size * priority) ** (-self.per_beta)
        is_weights /= is_weights.max()

        states = self.states[idx].reshape((nr_batches, nr_samples) + self.os_shape)
        next_states = self.next_states[idx].reshape((nr_batches, nr_samples) + self.os_shape)
        actions = self.actions[idx].reshape((nr_batches, nr_samples) + self.as_shape)
        rewards = self.rewards[idx].reshape((nr_batches, nr_samples))
        dones = self.dones[idx].reshape((nr_batches, nr_samples))
        is_weights = is_weights.reshape((nr_batches, nr_samples))
        idx = idx.reshape((nr_batches, nr_samples))

        return states, next_states, actions, rewards, dones, is_weights, idx
    

    def update_priorities(self, idx, td_error):
        priority = (np.abs(td_error) + self.per_epsilon) ** self.per_alpha
        self.highest_priority = max(self.highest_priority, priority.max())
        self.priority_tree.update(idx, priority)
