import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return torch.compile(VectorCritic(env, config.algorithm.nr_hidden_units, device, critic_observation_indices).to(device), mode=config.algorithm.compile_mode)


class Critic(nn.Module):
    def __init__(self, env, nr_hidden_units, device, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        self.network = nn.Sequential(
            nn.Linear(len(critic_observation_indices) + np.prod(env.single_action_space.shape, dtype=int).item(), nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1),
        )


    def forward(self, x, action):
        return self.network(torch.cat([x[..., self.critic_observation_indices], action], dim=-1))


class VectorCritic(nn.Module):
    def __init__(self, env, nr_hidden_units, device, critic_observation_indices):
        super().__init__()
        self.critics = nn.ModuleList([Critic(env, nr_hidden_units, device, critic_observation_indices) for _ in range(2)])


    def forward(self, x, action):
        return torch.stack([critic(x, action) for critic in self.critics])
