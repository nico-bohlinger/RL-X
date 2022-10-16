from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from rl_x.environments.action_space_type import ActionSpaceType


class Actor(nn.Module):
    def __init__(self, single_os_shape, single_as_shape, std_dev, nr_hidden_layers, nr_hidden_units, action_space_type):
        super().__init__()
        self.action_space_type = action_space_type

        if nr_hidden_layers < 0:
            raise ValueError("nr_hidden_layers must be >= 0")
        if nr_hidden_units < 1:
            raise ValueError("nr_hidden_units must be >= 1")
        
        
        if nr_hidden_layers == 0:
            self.actor_mean = nn.Sequential(self.layer_init(nn.Linear(single_os_shape, single_as_shape), std=0.01))
        else:
            layers = []
            layers.extend([
                (f"fc_{len(layers) + 1}", self.layer_init(nn.Linear(single_os_shape, nr_hidden_units))),
                (f"tanh_{len(layers) + 1}", nn.Tanh())
            ])
            for _ in range(nr_hidden_layers - 1):
                layers.extend([
                    (f"fc_{int(len(layers) / 2) + 1}", self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units))),
                    (f"tanh_{int(len(layers) / 2) + 1}", nn.Tanh())
                ])
            layers.append((f"fc_{int(len(layers) / 2) + 1}", self.layer_init(nn.Linear(nr_hidden_units, single_as_shape), std=0.01)))
            self.actor_mean = nn.Sequential(OrderedDict(layers))

        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            self.actor_logstd = nn.Parameter(torch.full((1, single_as_shape), np.log(std_dev)))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def get_action(self, x, action=None):
        action_mean = self.actor_mean(x)
        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            action_logstd = self.actor_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)
        elif self.action_space_type == ActionSpaceType.DISCRETE:
            probs = Categorical(logits=action_mean)
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action)
            entropy = probs.entropy()
        else:
            raise ValueError("Invalid action_space_type")
        
        return action, log_prob, entropy
