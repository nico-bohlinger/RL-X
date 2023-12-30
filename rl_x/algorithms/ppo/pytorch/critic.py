import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return FlatValuesCritic(env, config.algorithm.nr_hidden_units)
    elif observation_space_type == ObservationSpaceType.IMAGES:
        return ImagesCritic(env)


class FlatValuesCritic(nn.Module):
    def __init__(self, env, nr_hidden_units):
        super().__init__()
        single_os_shape = env.single_observation_space.shape

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape, dtype=int).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, 1), std=1.0),
        )


    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    @torch.compile(mode="default")
    def get_value(self, x):
        return self.critic(x)


class ImagesCritic(nn.Module):
    def __init__(self, env):
        super().__init__()
        single_os_shape = env.single_observation_space.shape

        self.critic = nn.Sequential(
            self.layer_init(nn.Conv2d(single_os_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, 1), std=1.0)
        )
        self.critic(torch.zeros(1, *single_os_shape))  # Init the lazy linear layer

    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    @torch.compile(mode="default")
    def get_value(self, x):
        if x.ndim == 5:
            x_shaped = x.reshape(-1, *x.shape[2:])
            return self.critic(x_shaped / 255.0).reshape(x.shape[0], x.shape[1], 1)
        return self.critic(x / 255.0)
