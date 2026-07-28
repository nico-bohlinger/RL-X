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
        return ScorePolicy(config, env, device, policy_observation_indices).to(device)


class ScorePolicy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        action_dimension = np.prod(env.single_action_space.shape, dtype=int).item()
        timestep_embed_dim = config.algorithm.timestep_embed_dim
        self.register_buffer("policy_observation_indices", torch.tensor(policy_observation_indices, dtype=torch.long, device=device))
        self.register_buffer("timestep_coefficients", torch.linspace(0.1, 100.0, timestep_embed_dim, device=device)[None])
        self.log_timestep = nn.Parameter(torch.full((1,), np.log(np.expm1(config.algorithm.initial_timestep))))
        self.log_friction = nn.Parameter(torch.full((action_dimension,), np.log(np.expm1(config.algorithm.initial_friction))))
        self.timestep_phase = nn.Parameter(torch.zeros((1, timestep_embed_dim)))
        self.timestep_dense_1 = nn.Linear(2 * timestep_embed_dim, timestep_embed_dim)
        self.timestep_dense_2 = nn.Linear(timestep_embed_dim, timestep_embed_dim)
        layers = []
        input_dimension = action_dimension + len(policy_observation_indices) + timestep_embed_dim
        for hidden_dimension in config.algorithm.score_hidden_dims:
            layers.extend([nn.Linear(input_dimension, hidden_dimension), nn.GELU()])
            input_dimension = hidden_dimension
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dimension, action_dimension)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1.0 / np.sqrt(module.in_features))
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.output_layer.weight, config.algorithm.score_output_scale)


    def forward(self, observation, action, timestep):
        observation = observation[..., self.policy_observation_indices]
        phase = self.timestep_phase.to(timestep.dtype)
        coefficients = self.timestep_coefficients.to(timestep.dtype)
        timestep_embedding = torch.cat([torch.sin(coefficients * timestep + phase), torch.cos(coefficients * timestep + phase)], dim=-1)
        timestep_embedding = self.timestep_dense_2(F.gelu(self.timestep_dense_1(timestep_embedding)))
        return torch.clamp(self.output_layer(self.network(torch.cat([action, observation, timestep_embedding], dim=-1))), -1e4, 1e4)
