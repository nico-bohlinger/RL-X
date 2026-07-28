import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return ValueCritic(config, device, critic_observation_indices).to(device)


class ValueCritic(nn.Module):
    def __init__(self, config, device, critic_observation_indices):
        super().__init__()
        hidden_dims = config.algorithm.critic_hidden_dims
        self.register_buffer("critic_observation_indices", torch.tensor(critic_observation_indices, dtype=torch.long, device=device))
        self.input_layer = nn.Linear(len(critic_observation_indices), hidden_dims[0])
        self.residual_layers = nn.ModuleList()
        for hidden_index in range(1, len(hidden_dims), 2):
            self.residual_layers.append(nn.ModuleList([nn.Linear(hidden_dims[hidden_index - 1], hidden_dims[hidden_index]), nn.Linear(hidden_dims[hidden_index], hidden_dims[hidden_index + 1])]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, -np.sqrt(3.0 / module.in_features), np.sqrt(3.0 / module.in_features))
                nn.init.zeros_(module.bias)


    def forward(self, observation):
        x = self.input_layer(observation[..., self.critic_observation_indices])
        for layer_1, layer_2 in self.residual_layers:
            residual = x
            x = x * torch.tanh(F.softplus(x))
            x = layer_1(x)
            x = x * torch.tanh(F.softplus(x))
            x = layer_2(x)
            x = x + residual
        return self.output_layer(x)
