import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, capacity, os_shape, as_shape, device):
        self.obs = np.zeros((capacity,) + os_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((capacity), dtype=np.float32)
        self.dones = np.zeros((capacity), dtype=np.float32)
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.device = device
    

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray):
        self.obs[self.pos] = obs.copy()
        self.next_obs[self.pos] = next_obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward.copy()
        self.dones[self.pos] = done.copy()
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    

    def sample(self, n):
        indices = np.random.randint(self.size, size=n)
        obs = torch.tensor(self.obs[indices], dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(self.next_obs[indices], dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.int64).to(self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device)
        return obs, next_obs, actions, rewards, dones
