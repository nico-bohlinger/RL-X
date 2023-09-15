import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.get_observation_space_type()

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return QNetwork(env, config.algorithm.nr_hidden_units)
    else:
        raise ValueError(f"Unsupported observation_space_type: {observation_space_type}")


class QNetwork(nn.Module):
    def __init__(self, env, nr_hidden_units):
        super().__init__()
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        self.critic = nn.Sequential(
            nn.Linear((np.prod(single_os_shape, dtype=int) + np.prod(single_as_shape, dtype=int)).item(), nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1),
        )


    def forward(self, x, a):
        return self.critic(torch.cat([x, a], dim=1))
