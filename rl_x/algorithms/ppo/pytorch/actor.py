import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Actor(nn.Module):
    def __init__(self, os_shape, as_shape, std_dev):
        super().__init__()
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(os_shape, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, as_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, as_shape), np.log(std_dev)))
    

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
