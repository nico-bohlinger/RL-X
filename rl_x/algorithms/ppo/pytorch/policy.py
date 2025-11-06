import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return ContinuousFlatValuesPolicy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.nr_hidden_units, device, policy_observation_indices)
    elif action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DiscreteFlatValuesPolicy(env, config.algorithm.nr_hidden_units, policy_observation_indices)
    elif action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.IMAGES:
        return ContinuousImagesPolicy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, device)
    elif action_space_type == ActionSpaceType.DISCRETE and observation_space_type == ObservationSpaceType.IMAGES:
        return DiscreteImagesPolicy(env)


class ContinuousFlatValuesPolicy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.policy_observation_indices = policy_observation_indices
        obs_input_dim = len(policy_observation_indices)
        self.policy_as_low = -1
        self.policy_as_high = 1
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32).to(device)
        single_as_shape = env.single_action_space.shape

        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Linear(obs_input_dim, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, np.prod(single_as_shape, dtype=int).item()), std=0.01),
        )
        self.policy_logstd = nn.Parameter(torch.full((1, np.prod(single_as_shape, dtype=int).item()), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    @torch.compile(mode="default")
    def get_action_logprob(self, x):
        x = x[..., self.policy_observation_indices]
        action_mean = self.policy_mean(x)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        if self.action_clipping_and_rescaling:
            clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
            clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        else:
            clipped_and_scaled_action = action
        return action, clipped_and_scaled_action, probs.log_prob(action).sum(1)
    

    @torch.compile(mode="default")
    def get_logprob_entropy(self, x, action):
        x = x[..., self.policy_observation_indices]
        action_mean = self.policy_mean(x)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)
    

    @torch.compile(mode="default")
    def get_deterministic_action(self, x):
        x = x[..., self.policy_observation_indices]
        with torch.no_grad():
            action = self.policy_mean(x)
        if self.action_clipping_and_rescaling:
            clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
            clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        else:
            clipped_and_scaled_action = action
        return clipped_and_scaled_action


class DiscreteFlatValuesPolicy(nn.Module):
    def __init__(self, env, nr_hidden_units, policy_observation_indices):
        super().__init__()
        self.policy_observation_indices = policy_observation_indices
        obs_input_dim = len(policy_observation_indices)
        logit_size = env.get_single_action_logit_size()
        
        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Linear(obs_input_dim, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, logit_size), std=0.01),
        )
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    @torch.compile(mode="default")
    def get_action_logprob(self, x):
        x = x[..., self.policy_observation_indices]
        action_mean = self.policy_mean(x)
        probs = Categorical(logits=action_mean)
        action = probs.sample()
        return action, action, probs.log_prob(action)
    

    @torch.compile(mode="default")
    def get_logprob_entropy(self, x, action):
        x = x[..., self.policy_observation_indices]
        action_mean = self.policy_mean(x)
        probs = Categorical(logits=action_mean)
        return probs.log_prob(action), probs.entropy()
    

    @torch.compile(mode="default")
    def get_deterministic_action(self, x):
        x = x[..., self.policy_observation_indices]
        with torch.no_grad():
            return self.policy_mean(x).argmax(1)


class ContinuousImagesPolicy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, device):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.policy_as_low = -1
        self.policy_as_high = 1
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32).to(device)
        single_os_shape = env.single_observation_space.shape
        single_as_shape = env.single_action_space.shape

        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Conv2d(single_os_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, np.prod(single_as_shape, dtype=int).item()), std=0.01)
        )
        self.policy_mean(torch.zeros(1, *single_os_shape))  # Init the lazy linear layer
        self.policy_logstd = nn.Parameter(torch.full((1, np.prod(single_as_shape, dtype=int).item()), np.log(std_dev)))
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    @torch.compile(mode="default")
    def get_action_logprob(self, x):
        action_mean = self.policy_mean(x / 255.0)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        if self.action_clipping_and_rescaling:
            clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
            clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        else:
            clipped_and_scaled_action = action
        return action, clipped_and_scaled_action, probs.log_prob(action).sum(1)
    

    @torch.compile(mode="default")
    def get_logprob_entropy(self, x, action):
        action_mean = self.policy_mean(x / 255.0)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)
    

    @torch.compile(mode="default")
    def get_deterministic_action(self, x):
        with torch.no_grad():
            action = self.policy_mean(x / 255.0)
        if self.action_clipping_and_rescaling:
            clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
            clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        else:
            clipped_and_scaled_action = action
        return clipped_and_scaled_action


class DiscreteImagesPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        single_os_shape = env.single_observation_space.shape
        logit_size = env.get_single_action_logit_size()

        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Conv2d(single_os_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            self.layer_init(nn.Linear(512, logit_size), std=0.01)
        )
        self.policy_mean(torch.zeros(1, *single_os_shape))  # Init the lazy linear layer
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    @torch.compile(mode="default")
    def get_action_logprob(self, x):
        action_mean = self.policy_mean(x / 255.0)
        probs = Categorical(logits=action_mean)
        action = probs.sample()
        return action, action, probs.log_prob(action)
    

    @torch.compile(mode="default")
    def get_logprob_entropy(self, x, action):
        action_mean = self.policy_mean(x / 255.0)
        probs = Categorical(logits=action_mean)
        return probs.log_prob(action), probs.entropy()
    

    @torch.compile(mode="default")
    def get_deterministic_action(self, x):
        with torch.no_grad():
            return self.policy_mean(x / 255.0).argmax(1)
