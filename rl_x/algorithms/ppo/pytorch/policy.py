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
        return ContinuousFlatValuesPolicy(env, config.algorithm.std_dev, config.algorithm.nr_hidden_units, device)
    elif action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DiscreteFlatValuesPolicy(env, config.algorithm.nr_hidden_units)
    elif action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.IMAGES:
        return ContinuousImagesPolicy(env, config.algorithm.std_dev, device)
    elif action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.IMAGES:
        return DiscreteImagesPolicy(env)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class ContinuousFlatValuesPolicy(nn.Module):
    def __init__(self, env, std_dev, nr_hidden_units, device):
        super().__init__()
        self.policy_as_low = -1
        self.policy_as_high = 1
        self.env_as_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()
        
        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, np.prod(single_as_shape).item()), std=0.01),
        )
        self.policy_logstd = nn.Parameter(torch.full((1, np.prod(single_as_shape).item()), np.log(std_dev).item()))
    

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


class DiscreteFlatValuesPolicy(nn.Module):
    def __init__(self, env, nr_hidden_units):
        super().__init__()
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()
        
        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, np.prod(single_as_shape).item()), std=0.01),
        )
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_action_logprob(self, x):
        action_mean = self.policy_mean(x)
        probs = Categorical(logits=action_mean)
        action = probs.sample()
        return action, action, probs.log_prob(action)
    

    def get_logprob_entropy(self, x, action=None):
        action_mean = self.policy_mean(x)
        probs = Categorical(logits=action_mean)
        return probs.log_prob(action), probs.entropy()
    

    def get_deterministic_action(self, x):
        with torch.no_grad():
            return self.policy_mean(x).argmax(1)


class ContinuousImagesPolicy(nn.Module):
    def __init__(self, env, std_dev, device):
        super().__init__()
        self.policy_as_low = -1
        self.policy_as_high = 1
        self.env_as_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        single_as_shape = env.get_single_action_space_shape()

        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Conv2d(env.observation_space.shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, np.prod(single_as_shape)), std=0.01)
        )
        self.policy_mean(torch.zeros(1, *env.observation_space.shape))  # Init the lazy linear layer
        self.policy_logstd = nn.Parameter(torch.full((1, np.prod(single_as_shape)), np.log(std_dev)))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def get_action_logprob(self, x):
        action_mean = self.policy_mean(x / 255.0)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
        clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        return action, clipped_and_scaled_action, probs.log_prob(action).sum(1)
    

    def get_logprob_entropy(self, x, action):
        action_mean = self.policy_mean(x / 255.0)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)
    

    def get_deterministic_action(self, x):
        with torch.no_grad():
            action = self.policy_mean(x / 255.0)
        clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
        clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        return clipped_and_scaled_action


class DiscreteImagesPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        single_as_shape = env.get_single_action_space_shape()

        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Conv2d(env.observation_space.shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, np.prod(single_as_shape)), std=0.01)
        )
        self.policy_mean(torch.zeros(1, *env.observation_space.shape))  # Init the lazy linear layer
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def get_action_logprob(self, x):
        action_mean = self.policy_mean(x / 255.0)
        probs = Categorical(logits=action_mean)
        action = probs.sample()
        return action, action, probs.log_prob(action)
    

    def get_logprob_entropy(self, x, action=None):
        action_mean = self.policy_mean(x / 255.0)
        probs = Categorical(logits=action_mean)
        return probs.log_prob(action), probs.entropy()
    

    def get_deterministic_action(self, x):
        with torch.no_grad():
            return self.policy_mean(x / 255.0).argmax(1)
