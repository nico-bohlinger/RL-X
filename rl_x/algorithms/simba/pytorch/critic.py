import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.simba.pytorch.layers import SimbaEncoder


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return torch.compile(VectorCritic(config, env, device).to(device), mode=config.algorithm.compile_mode)


class Critic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        input_dim = len(critic_observation_indices) + np.prod(env.single_action_space.shape, dtype=int).item()
        self.encoder = SimbaEncoder(input_dim, config.algorithm.critic_hidden_dim, config.algorithm.critic_nr_blocks)
        self.value = nn.Linear(config.algorithm.critic_hidden_dim, 1)
        nn.init.orthogonal_(self.value.weight, 1.0)
        nn.init.zeros_(self.value.bias)


    def forward(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        return self.value(self.encoder(torch.cat([obs, action], dim=-1))).squeeze(-1)


class VectorCritic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        nr_critics = 2 if config.algorithm.use_cdq else 1
        self.critics = nn.ModuleList([Critic(config, env, device) for _ in range(nr_critics)])


    def forward(self, obs, action):
        values = torch.stack([critic(obs, action) for critic in self.critics])
        return values if len(self.critics) == 2 else values[0]
