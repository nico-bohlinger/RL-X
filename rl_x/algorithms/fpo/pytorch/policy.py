import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return FlowPolicy(config, env, device, policy_observation_indices).to(device)


class FlowPolicy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.timestep_embed_dim = config.algorithm.timestep_embed_dim
        self.output_scale = config.algorithm.policy_output_scale
        action_dimension = np.prod(env.single_action_space.shape, dtype=int).item()
        self.register_buffer("policy_observation_indices", torch.tensor(policy_observation_indices, dtype=torch.long, device=device))
        self.register_buffer("frequencies", 2 ** torch.arange(self.timestep_embed_dim // 2, dtype=torch.float32, device=device))
        layers = []
        input_dimension = len(policy_observation_indices) + self.timestep_embed_dim + action_dimension
        for hidden_dimension in config.algorithm.policy_hidden_dims:
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
