import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = torch.compile(Policy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.nr_hidden_units, device, policy_observation_indices).to(device), mode=compile_mode)
        policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=compile_mode)
        policy.get_logprob_entropy = torch.compile(policy.get_logprob_entropy, mode=compile_mode)
        policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=compile_mode)
        return policy


class Policy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
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
    

    def get_logprob_entropy(self, x, action):
        x = x[..., self.policy_observation_indices]
        action_mean = self.policy_mean(x)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)
    

    def get_deterministic_action(self, x):
        x = x[..., self.policy_observation_indices]
        action = self.policy_mean(x)
        if self.action_clipping_and_rescaling:
            clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
            clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        else:
            clipped_and_scaled_action = action
        return clipped_and_scaled_action
