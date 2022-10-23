import numpy as np
import jax.numpy as jnp


class ReplayBuffer():
    def __init__(self, capacity, nr_envs, os_shape, as_shape):
        self.capacity = capacity // nr_envs
        self.nr_envs = nr_envs
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0
    

    def add(self, states, next_states, actions, rewards, dones):
        self.states[self.pos] = states.copy()
        self.next_states[self.pos] = next_states.copy()
        self.actions[self.pos] = actions.copy()
        self.rewards[self.pos] = rewards.copy()
        self.dones[self.pos] = dones.copy()
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, n):
        idx1 = np.random.randint(self.size, size=n)
        idx2 = np.random.randint(self.nr_envs, size=n)
        states = self.states[idx1, idx2]
        next_states = self.next_states[idx1, idx2]
        actions = self.actions[idx1, idx2]
        rewards = self.rewards[idx1, idx2]
        dones = self.dones[idx1, idx2]
        return states, next_states, actions, rewards, dones
