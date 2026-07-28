import math
import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.simbav2.pytorch.layers import HyperDense, HyperEmbedder, HyperLERPBlock, Scaler


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = torch.compile(Policy(config, env, device, policy_observation_indices).to(device), mode=config.algorithm.compile_mode)
        policy.forward = torch.compile(policy.forward, mode=config.algorithm.compile_mode)
        policy.get_action = torch.compile(policy.get_action, mode=config.algorithm.compile_mode)
        policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
        return policy


class Policy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.log_std_min = config.algorithm.log_std_min
        self.log_std_max = config.algorithm.log_std_max
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()
        hidden_dim = config.algorithm.policy_hidden_dim
        scaler_init = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (config.algorithm.policy_nr_blocks + 1)

        self.embedder = HyperEmbedder(len(policy_observation_indices), hidden_dim, scaler_init, scaler_init, config.algorithm.c_shift)
        self.blocks = nn.ModuleList([HyperLERPBlock(hidden_dim, scaler_init, scaler_init, alpha_init, 1.0 / math.sqrt(hidden_dim)) for _ in range(config.algorithm.policy_nr_blocks)])
        self.mean_hidden = HyperDense(hidden_dim, hidden_dim)
        self.mean_scaler = Scaler(hidden_dim, 1.0, 1.0)
        self.mean = HyperDense(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.log_std_hidden = HyperDense(hidden_dim, hidden_dim)
        self.log_std_scaler = Scaler(hidden_dim, 1.0, 1.0)
        self.log_std = HyperDense(hidden_dim, action_dim)
        self.log_std_bias = nn.Parameter(torch.zeros(action_dim))


    def forward(self, x):
        x = self.embedder(x[..., self.policy_observation_indices])
        for block in self.blocks:
            x = block(x)
        mean = self.mean(self.mean_scaler(self.mean_hidden(x)))
        mean = mean + self.mean_bias.to(mean.dtype)
        log_std = self.log_std(self.log_std_scaler(self.log_std_hidden(x)))
        log_std = log_std + self.log_std_bias.to(log_std.dtype)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1.0 + torch.tanh(log_std))
        return mean, log_std


    def get_action(self, x):
        mean, log_std = self(x)
        noise = torch.randn_like(mean)
        pretanh = mean + torch.exp(log_std) * noise
        action = torch.tanh(pretanh)
        gaussian_log_prob = -0.5 * noise ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std
        tanh_correction = 2.0 * (np.log(2.0) - pretanh - torch.nn.functional.softplus(-2.0 * pretanh))
        log_prob = (gaussian_log_prob - tanh_correction).sum(dim=-1)
        processed_action = self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
        return action, processed_action, log_prob


    def get_deterministic_action(self, x):
        mean, _ = self(x)
        action = torch.tanh(mean)
        return self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
