from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_actor(config, env):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return ContinuousFlatValuesActor(env, config.algorithm.std_dev, config.algorithm.nr_hidden_layers, config.algorithm.nr_hidden_units)
    elif action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DiscreteFlatValuesActor(env, config.algorithm.nr_hidden_layers, config.algorithm.nr_hidden_units)
    elif action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.IMAGES:
        return ContinuousImagesActor(env, config.algorithm.std_dev)
    elif action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.IMAGES:
        return DiscreteImagesActor(env)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class ContinuousFlatValuesActor(nn.Module):
    def __init__(self, env, std_dev, nr_hidden_layers, nr_hidden_units):
        super().__init__()
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        if nr_hidden_layers < 0:
            raise ValueError("nr_hidden_layers must be >= 0")
        if nr_hidden_units < 1:
            raise ValueError("nr_hidden_units must be >= 1")
        
        if nr_hidden_layers == 0:
            self.actor_mean = nn.Sequential(self.layer_init(nn.Linear(np.prod(single_os_shape), single_as_shape), std=0.01))
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
            layers.append((f"fc_{int(len(layers) / 2) + 1}", self.layer_init(nn.Linear(nr_hidden_units, single_as_shape), std=0.01)))
            self.actor_mean = nn.Sequential(OrderedDict(layers))

        self.actor_logstd = nn.Parameter(torch.full((1, single_as_shape), np.log(std_dev)))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)



class DiscreteFlatValuesActor(nn.Module):
    def __init__(self, env, nr_hidden_layers, nr_hidden_units):
        super().__init__()
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        if nr_hidden_layers < 0:
            raise ValueError("nr_hidden_layers must be >= 0")
        if nr_hidden_units < 1:
            raise ValueError("nr_hidden_units must be >= 1")
        
        if nr_hidden_layers == 0:
            self.actor_mean = nn.Sequential(self.layer_init(nn.Linear(np.prod(single_os_shape), single_as_shape), std=0.01))
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
            layers.append((f"fc_{int(len(layers) / 2) + 1}", self.layer_init(nn.Linear(nr_hidden_units, single_as_shape), std=0.01)))
            self.actor_mean = nn.Sequential(OrderedDict(layers))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        probs = Categorical(logits=action_mean)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()


class ContinuousImagesActor(nn.Module):
    def __init__(self, env, std_dev):
        super().__init__()
        single_as_shape = env.get_single_action_space_shape()

        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Conv2d(env.observation_space.shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, single_as_shape), std=0.01)
        )
        self.actor_mean(torch.zeros(1, *env.observation_space.shape))  # Init the lazy linear layer
        self.actor_logstd = nn.Parameter(torch.full((1, single_as_shape), np.log(std_dev)))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x / 255.0)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)


class DiscreteImagesActor(nn.Module):
    def __init__(self, env):
        super().__init__()
        single_as_shape = env.get_single_action_space_shape()

        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Conv2d(env.observation_space.shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, single_as_shape), std=0.01)
        )
        self.actor_mean(torch.zeros(1, *env.observation_space.shape))  # Init the lazy linear layer
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x / 255.0)
        probs = Categorical(logits=action_mean)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
