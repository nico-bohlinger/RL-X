import torch
import torch.nn as nn


class FlowPolicy(nn.Module):
    def __init__(self, action_dimension, timestep_embed_dim, hidden_dims, output_scale, policy_observation_indices, device):
        super().__init__()
        self.timestep_embed_dim = timestep_embed_dim
        self.output_scale = output_scale
        self.register_buffer("policy_observation_indices", torch.tensor(policy_observation_indices, dtype=torch.long, device=device))
        self.register_buffer("frequencies", 2 ** torch.arange(timestep_embed_dim // 2, dtype=torch.float32, device=device))
        layers = []
        input_dimension = len(policy_observation_indices) + timestep_embed_dim + action_dimension
        for hidden_dimension in hidden_dims:
            layers.extend([nn.Linear(input_dimension, hidden_dimension), nn.ELU()])
            input_dimension = hidden_dimension
        layers.append(nn.Linear(input_dimension, action_dimension))
        self.network = nn.Sequential(*layers)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)


    def forward(self, observation, noisy_action, timestep):
        observation = observation[..., self.policy_observation_indices]
        scaled_timestep = timestep * self.frequencies.to(timestep.dtype)
        timestep_embedding = torch.cat([torch.cos(scaled_timestep), torch.sin(scaled_timestep)], dim=-1)
        return self.network(torch.cat([observation, timestep_embedding, noisy_action], dim=-1)) * self.output_scale


class ValueCritic(nn.Module):
    def __init__(self, hidden_dims, critic_observation_indices, device):
        super().__init__()
        self.register_buffer("critic_observation_indices", torch.tensor(critic_observation_indices, dtype=torch.long, device=device))
        layers = []
        input_dimension = len(critic_observation_indices)
        for hidden_dimension in hidden_dims:
            layers.extend([nn.Linear(input_dimension, hidden_dimension), nn.ELU()])
            input_dimension = hidden_dimension
        layers.append(nn.Linear(input_dimension, 1))
        self.network = nn.Sequential(*layers)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)


    def forward(self, observation):
        return self.network(observation[..., self.critic_observation_indices])
