import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


class QNetwork(nn.Module):
    def __init__(self, env, nr_hidden_units, nr_heads_per_net, device, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        single_as_shape = env.single_action_space.shape
        obs_input_dim = len(critic_observation_indices)
        action_input_dim = np.prod(single_as_shape, dtype=int).item()

        self.critic = nn.Sequential(
            nn.Linear(obs_input_dim + action_input_dim, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_heads_per_net),
        )


    def forward(self, x, a):
        x = x[..., self.critic_observation_indices]
        return self.critic(torch.cat([x, a], dim=1))


class VectorQNetwork(nn.Module):
    def __init__(self, config, env, device, critic_observation_indices):
        super().__init__()
        self.critics = nn.ModuleList([QNetwork(env, config.algorithm.nr_hidden_units, config.algorithm.nr_heads_per_net, device, critic_observation_indices) for _ in range(config.algorithm.ensemble_size)])


    def forward(self, x, action):
        return torch.stack([critic(x, action) for critic in self.critics])
