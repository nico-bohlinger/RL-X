import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        self.critic = nn.Sequential(nn.Linear(len(critic_observation_indices), 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.critic[0].weight, np.sqrt(2.0))
        nn.init.zeros_(self.critic[0].bias)
        nn.init.orthogonal_(self.critic[2].weight, np.sqrt(2.0))
        nn.init.zeros_(self.critic[2].bias)
        nn.init.orthogonal_(self.critic[4].weight, 1.0)
        nn.init.zeros_(self.critic[4].bias)


    def forward(self, x):
        return self.critic(x[..., self.critic_observation_indices])


    def get_value(self, x):
        return self(x)
