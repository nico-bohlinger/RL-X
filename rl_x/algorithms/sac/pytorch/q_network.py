import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_q_network(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return QNetwork(env, config.algorithm.gamma, config.algorithm.nr_hidden_units)


class QNetwork(nn.Module):
    def __init__(self, env, gamma, nr_hidden_units):
        super().__init__()
        self.gamma = gamma
        single_os_shape = env.single_observation_space.shape
        single_as_shape = env.single_action_space.shape

        self.critic = nn.Sequential(
            nn.Linear((np.prod(single_os_shape, dtype=int) + np.prod(single_as_shape, dtype=int)).item(), nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1),
        )

    
    @torch.compile(mode="default")
    def forward(self, x, a):
        return self.critic(torch.cat([x, a], dim=1))
