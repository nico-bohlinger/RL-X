from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.get_observation_space_type()

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return FlatValuesCritic(env, config.algorithm.nr_hidden_layers, config.algorithm.nr_hidden_units)
    elif observation_space_type == ObservationSpaceType.IMAGES:
        return ImagesCritic(env)
    else:
        raise ValueError(f"Unsupported observation_space_type: {observation_space_type}")


class FlatValuesCritic(nn.Module):
    def __init__(self, env, nr_hidden_layers, nr_hidden_units):
        super().__init__()
        single_os_shape = env.observation_space.shape

        if nr_hidden_layers < 0:
            raise ValueError("nr_hidden_layers must be >= 0")
        if nr_hidden_units < 1:
            raise ValueError("nr_hidden_units must be >= 1")

        if nr_hidden_layers == 0:
            self.critic = nn.Sequential(self.layer_init(nn.Linear(np.prod(single_os_shape), 1), std=1.0))
        else:
            layers = []
            layers.extend([
                (f"fc_{len(layers) + 1}", self.layer_init(nn.Linear(np.prod(single_os_shape), nr_hidden_units))),
                (f"tanh_{len(layers) + 1}", nn.Tanh())
            ])
            for _ in range(nr_hidden_layers - 1):
                layers.extend([
                    (f"fc_{int(len(layers) / 2) + 1}", self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units))),
                    (f"tanh_{int(len(layers) / 2) + 1}", nn.Tanh())
                ])
            layers.append((f"fc_{int(len(layers) / 2) + 1}", self.layer_init(nn.Linear(nr_hidden_units, 1), std=1.0)))
            self.critic = nn.Sequential(OrderedDict(layers))

    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_value(self, x):
        return self.critic(x)


class ImagesCritic(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.critic = nn.Sequential(
            self.layer_init(nn.Conv2d(env.observation_space.shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, 1), std=0.01)
        )
        self.critic(torch.zeros(1, *env.observation_space.shape))  # Init the lazy linear layer

    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_value(self, x):
        return self.critic(x / 255.0)
