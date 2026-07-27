import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.bro.pytorch.layers import BroNet


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return torch.compile(DoubleCritic(config, env, device).to(device), mode=config.algorithm.compile_mode)


class SingleCritic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()
        output_nodes = config.algorithm.nr_quantiles if config.algorithm.distributional else 1
        self.net = BroNet(len(critic_observation_indices) + action_dim, config.algorithm.critic_hidden_dim, config.algorithm.critic_nr_blocks, output_nodes)


    def forward(self, obs, action):
        q = self.net(torch.cat([obs[..., self.critic_observation_indices], action], dim=-1))
        return q.squeeze(-1) if q.shape[-1] == 1 else q


class DoubleCritic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.q1 = SingleCritic(config, env, device)
        self.q2 = SingleCritic(config, env, device)


    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)
