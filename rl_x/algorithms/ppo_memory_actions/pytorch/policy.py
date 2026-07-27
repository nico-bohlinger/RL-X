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
    policy_observation_indices = np.concatenate([policy_observation_indices, np.arange(env.single_observation_space.shape[0], env.single_observation_space.shape[0] + config.algorithm.memory_action_dimension)])

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = Policy(env, config.algorithm.memory_action_dimension, config.algorithm.memory_action_mean_clip, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.nr_hidden_units, device, policy_observation_indices).to(device)

    policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
    policy.get_logprob_entropy = torch.compile(policy.get_logprob_entropy, mode=config.algorithm.compile_mode)
    policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
    return policy


class Policy(nn.Module):
    def __init__(self, env, memory_action_dimension, memory_action_mean_clip, std_dev, action_clipping_and_rescaling, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.memory_action_dimension = memory_action_dimension
        self.memory_action_mean_clip = memory_action_mean_clip
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.env_action_dimension = env.single_action_space.shape[0]
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Linear(len(policy_observation_indices), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, self.env_action_dimension + memory_action_dimension), std=0.01),
        )
        self.policy_logstd = nn.Parameter(torch.full((1, self.env_action_dimension + memory_action_dimension), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_mean_logstd(self, x):
        x = x[..., self.policy_observation_indices]
        env_action, memory_action = torch.split(self.policy_mean(x), [self.env_action_dimension, self.memory_action_dimension], dim=-1)
        memory_action = torch.clip(memory_action, -self.memory_action_mean_clip, self.memory_action_mean_clip)
        mean = torch.cat([env_action, memory_action], dim=-1)
        return mean, self.policy_logstd.expand_as(mean)


    def get_action_logprob(self, x):
        mean, logstd = self.get_mean_logstd(x)
        distribution = Normal(mean, logstd.exp())
        action = distribution.sample()
        env_action, memory_action = torch.split(action, [self.env_action_dimension, self.memory_action_dimension], dim=-1)
        if self.action_clipping_and_rescaling:
            processed_action = self.env_as_low + 0.5 * (torch.clip(env_action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        else:
            processed_action = env_action
        return action, processed_action, distribution.log_prob(action).sum(-1), memory_action / self.memory_action_mean_clip


    def get_logprob_entropy(self, x, action):
        mean, logstd = self.get_mean_logstd(x)
        distribution = Normal(mean, logstd.exp())
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


    def get_deterministic_action(self, x):
        action, logstd = self.get_mean_logstd(x)
        env_action, memory_action = torch.split(action, [self.env_action_dimension, self.memory_action_dimension], dim=-1)
        if self.action_clipping_and_rescaling:
            processed_action = self.env_as_low + 0.5 * (torch.clip(env_action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        else:
            processed_action = env_action
        return processed_action, memory_action / self.memory_action_mean_clip
