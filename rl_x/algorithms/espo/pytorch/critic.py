from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.get_observation_space_type()

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(env, config.algorithm.nr_hidden_units)
    else:
        raise ValueError(f"Unsupported observation_space_type: {observation_space_type}")


class Critic(nn.Module):
    def __init__(self, env, nr_hidden_units):
        super().__init__()
        single_os_shape = env.observation_space.shape

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, 1), std=1.0),
        )

    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_value(self, x):
        return self.critic(x)
