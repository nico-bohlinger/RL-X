from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Policy(env, config.algorithm.std_dev, config.algorithm.nr_hidden_layers, config.algorithm.nr_hidden_units, device)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class Policy(nn.Module):
    def __init__(self, env, std_dev, nr_hidden_layers, nr_hidden_units, device):
        super().__init__()
        self.policy_as_low = -1
        self.policy_as_high = 1
        self.env_as_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        if nr_hidden_layers < 0:
            raise ValueError("nr_hidden_layers must be >= 0")
        if nr_hidden_units < 1:
            raise ValueError("nr_hidden_units must be >= 1")
        
        if nr_hidden_layers == 0:
            self.policy_mean = nn.Sequential(self.layer_init(nn.Linear(np.prod(single_os_shape), np.prod(single_as_shape)), std=0.01))
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
            layers.append((f"fc_{int(len(layers) / 2) + 1}", self.layer_init(nn.Linear(nr_hidden_units, np.prod(single_as_shape)), std=0.01)))
            self.policy_mean = nn.Sequential(OrderedDict(layers))

        self.policy_logstd = nn.Parameter(torch.full((1, np.prod(single_as_shape)), np.log(std_dev)))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def get_action_logprob(self, x):
        action_mean = self.policy_mean(x)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
        clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        return action, clipped_and_scaled_action, probs.log_prob(action).sum(1)
    

    def get_logprob_entropy(self, x, action):
        action_mean = self.policy_mean(x)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)
    

    def get_deterministic_action(self, x):
        with torch.no_grad():
            action = self.policy_mean(x)
        clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
        clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        return clipped_and_scaled_action
