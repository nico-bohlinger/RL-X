import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DiffusionPolicy(config, env, device, policy_observation_indices).to(device)


class DiffusionPolicy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.output_scale = config.algorithm.policy_output_scale
        timestep_embed_dim = config.algorithm.timestep_embed_dim
        hidden_dims = config.algorithm.policy_hidden_dims
        action_dimension = np.prod(env.single_action_space.shape, dtype=int).item()
        self.register_buffer("policy_observation_indices", torch.tensor(policy_observation_indices, dtype=torch.long, device=device))
        self.register_buffer("frequencies", torch.exp(-np.log(10000.0) * torch.arange(timestep_embed_dim // 2, dtype=torch.float32, device=device) / (timestep_embed_dim // 2 - 1)))
        self.timestep_dense_1 = nn.Linear(timestep_embed_dim, 2 * timestep_embed_dim)
        self.timestep_dense_2 = nn.Linear(2 * timestep_embed_dim, timestep_embed_dim)
        self.input_layer = nn.Linear(action_dimension + timestep_embed_dim + len(policy_observation_indices), hidden_dims[0])
        self.residual_layers = nn.ModuleList()
        for hidden_index in range(1, len(hidden_dims), 2):
            self.residual_layers.append(nn.ModuleList([nn.Linear(hidden_dims[hidden_index - 1], hidden_dims[hidden_index]), nn.Linear(hidden_dims[hidden_index], hidden_dims[hidden_index + 1])]))
        self.output_layer = nn.Linear(hidden_dims[-1], action_dimension)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in [self.timestep_dense_1, self.timestep_dense_2]:
                    nn.init.normal_(module.weight, std=1.0 / np.sqrt(module.in_features))
                else:
                    nn.init.uniform_(module.weight, -np.sqrt(3.0 / module.in_features), np.sqrt(3.0 / module.in_features))
                nn.init.zeros_(module.bias)


    def forward(self, observation, noisy_action, timestep):
        observation = observation[..., self.policy_observation_indices]
        scaled_timestep = timestep * self.frequencies.to(timestep.dtype)
        timestep_embedding = torch.cat([torch.sin(scaled_timestep), torch.cos(scaled_timestep)], dim=-1)
        timestep_embedding = self.timestep_dense_1(timestep_embedding)
        timestep_embedding = timestep_embedding * torch.tanh(F.softplus(timestep_embedding))
        timestep_embedding = self.timestep_dense_2(timestep_embedding)
        x = self.input_layer(torch.cat([noisy_action, timestep_embedding, observation], dim=-1))
        for layer_1, layer_2 in self.residual_layers:
            residual = x
            x = layer_1(F.relu(x))
            x = layer_2(F.relu(x))
            x = x + residual
        return self.output_layer(x) * self.output_scale
