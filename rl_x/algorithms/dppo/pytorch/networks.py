import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionPolicy(nn.Module):
    def __init__(self, action_dimension, timestep_embed_dim, hidden_dims, output_scale, policy_observation_indices, device):
        super().__init__()
        self.output_scale = output_scale
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


class ValueCritic(nn.Module):
    def __init__(self, hidden_dims, critic_observation_indices, device):
        super().__init__()
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
