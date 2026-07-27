import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, nr_envs, observation_shape, action_shape, rng, device):
        self.buffer_size = buffer_size // nr_envs
        self.nr_envs = nr_envs
        self.rng = rng
        self.device = device
        self.states = np.zeros((self.buffer_size, nr_envs) + observation_shape, dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, nr_envs) + observation_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, nr_envs) + action_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, nr_envs), dtype=np.float32)
        self.terminations = np.zeros((self.buffer_size, nr_envs), dtype=np.float32)
        self.pos = 0
        self.full = False


    def add(self, states, next_states, actions, rewards, terminations):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.terminations[self.pos] = terminations
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        time_indices = self.rng.integers(0, upper_bound, size=batch_size)
        env_indices = self.rng.integers(0, self.nr_envs, size=batch_size)
        return (
            torch.tensor(self.states[time_indices, env_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_states[time_indices, env_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[time_indices, env_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.rewards[time_indices, env_indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.terminations[time_indices, env_indices], dtype=torch.float32, device=self.device),
        )
