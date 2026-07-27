import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.bro.pytorch.layers import BroNet


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        pessimistic_policy = torch.compile(NormalTanhPolicy(config, env, device, policy_observation_indices).to(device), mode=config.algorithm.compile_mode)
        optimistic_policy = torch.compile(DualTanhPolicy(config, env, device, policy_observation_indices).to(device), mode=config.algorithm.compile_mode)
        pessimistic_policy.forward = torch.compile(pessimistic_policy.forward, mode=config.algorithm.compile_mode)
        pessimistic_policy.get_action = torch.compile(pessimistic_policy.get_action, mode=config.algorithm.compile_mode)
        pessimistic_policy.get_deterministic_action = torch.compile(pessimistic_policy.get_deterministic_action, mode=config.algorithm.compile_mode)
        optimistic_policy.forward = torch.compile(optimistic_policy.forward, mode=config.algorithm.compile_mode)
        optimistic_policy.get_action = torch.compile(optimistic_policy.get_action, mode=config.algorithm.compile_mode)
        return pessimistic_policy, optimistic_policy


class NormalTanhPolicy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.log_std_min = -10.0
        self.log_std_max = 2.0
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()
        self.trunk = BroNet(len(policy_observation_indices), config.algorithm.policy_hidden_dim, config.algorithm.policy_nr_blocks)
        self.mean = nn.Linear(config.algorithm.policy_hidden_dim, action_dim)
        self.log_std = nn.Linear(config.algorithm.policy_hidden_dim, action_dim)
        nn.init.orthogonal_(self.mean.weight, np.sqrt(2.0))
        nn.init.zeros_(self.mean.bias)
        nn.init.orthogonal_(self.log_std.weight, 1.0)
        nn.init.zeros_(self.log_std.bias)


    def forward(self, obs):
        trunk = self.trunk(obs[..., self.policy_observation_indices])
        mean = self.mean(trunk)
        log_std = self.log_std(trunk)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1.0 + torch.tanh(log_std))
        return mean, torch.exp(log_std)


    def get_action(self, obs):
        mean, std = self(obs)
        noise = torch.randn_like(mean)
        pretanh = mean + std * noise
        action = torch.tanh(pretanh)
        gaussian_log_prob = -0.5 * noise ** 2 - 0.5 * np.log(2.0 * np.pi) - torch.log(std)
        tanh_correction = 2.0 * (np.log(2.0) - pretanh - torch.nn.functional.softplus(-2.0 * pretanh))
        log_prob = (gaussian_log_prob - tanh_correction).sum(dim=-1)
        processed_action = self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
        return action, processed_action, log_prob


    def get_deterministic_action(self, obs):
        mean, _ = self(obs)
        action = torch.tanh(mean)
        return self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)


class DualTanhPolicy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()
        self.trunk = BroNet(len(policy_observation_indices) + action_dim, config.algorithm.policy_hidden_dim, config.algorithm.policy_nr_blocks)
        self.shift = nn.Linear(config.algorithm.policy_hidden_dim, action_dim, bias=False)
        nn.init.orthogonal_(self.shift.weight, 0.01)


    def forward(self, obs, base_mean, base_std, std_multiplier):
        trunk = self.trunk(torch.cat([obs[..., self.policy_observation_indices], base_mean], dim=-1))
        return base_mean + self.shift(trunk), base_std * std_multiplier


    def get_action(self, obs, base_mean, base_std, std_multiplier):
        mean, std = self(obs, base_mean, base_std, std_multiplier)
        noise = torch.randn_like(mean)
        pretanh = mean + std * noise
        action = torch.tanh(pretanh)
        gaussian_log_prob = -0.5 * noise ** 2 - 0.5 * np.log(2.0 * np.pi) - torch.log(std)
        tanh_correction = 2.0 * (np.log(2.0) - pretanh - torch.nn.functional.softplus(-2.0 * pretanh))
        log_prob = (gaussian_log_prob - tanh_correction).sum(dim=-1)
        processed_action = self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
        return action, processed_action, log_prob, mean, std
