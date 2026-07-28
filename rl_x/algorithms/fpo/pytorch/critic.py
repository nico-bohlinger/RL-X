import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return ValueCritic(config, device, critic_observation_indices).to(device)


class ValueCritic(nn.Module):
    def __init__(self, config, device, critic_observation_indices):
        super().__init__()
        self.register_buffer("critic_observation_indices", torch.tensor(critic_observation_indices, dtype=torch.long, device=device))
        layers = []
        input_dimension = len(critic_observation_indices)
        for hidden_dimension in config.algorithm.critic_hidden_dims:
            layers.extend([nn.Linear(input_dimension, hidden_dimension), nn.ELU()])
            input_dimension = hidden_dimension
        layers.append(nn.Linear(input_dimension, 1))
        self.network = nn.Sequential(*layers)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)


    def forward(self, observation):
        return self.network(observation[..., self.critic_observation_indices])
