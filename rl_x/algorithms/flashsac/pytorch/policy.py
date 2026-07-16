import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.flashsac.pytorch.layers import FlashSACBlock, FlashSACEmbedder, UnitLinear, UnitRMSNorm


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = Policy(
            env=env,
            hidden_dim=config.algorithm.policy_hidden_dim,
            nr_blocks=config.algorithm.policy_nr_blocks,
            policy_observation_indices=policy_observation_indices,
            device=device,
        ).to(device)
        if config.algorithm.use_compile:
            policy.forward = torch.compile(policy.forward, mode=config.algorithm.compile_mode)
            policy.sample_and_log_prob = torch.compile(policy.sample_and_log_prob, mode=config.algorithm.compile_mode)
            policy.sample_with_noise = torch.compile(policy.sample_with_noise, mode=config.algorithm.compile_mode)
            policy.deterministic_action = torch.compile(policy.deterministic_action, mode=config.algorithm.compile_mode)
            policy.get_processed_action = torch.compile(policy.get_processed_action, mode=config.algorithm.compile_mode)
        return policy


class Policy(nn.Module):
    LOG_STD_MIN = -10.0
    LOG_STD_MAX = 2.0

    def __init__(self, env, hidden_dim, nr_blocks, policy_observation_indices, device):
        super().__init__()
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        nr_observations = len(policy_observation_indices)
        self.action_dim = np.prod(env.single_action_space.shape, dtype=int).item()
        env_as_low = torch.as_tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        env_as_high = torch.as_tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        if hasattr(env.single_action_space, "center") and hasattr(env.single_action_space, "scale"):
            env_as_center = torch.as_tensor(env.single_action_space.center, dtype=torch.float32, device=device)
            env_as_scale = torch.as_tensor(env.single_action_space.scale, dtype=torch.float32, device=device)
            env_as_low = (env_as_low - env_as_center) / env_as_scale
            env_as_high = (env_as_high - env_as_center) / env_as_scale
        self.register_buffer("env_as_low", env_as_low)
        self.register_buffer("env_as_high", env_as_high)

        self.embedder = FlashSACEmbedder(nr_observations, hidden_dim)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(nr_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.mean_w = UnitLinear(hidden_dim, self.action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(self.action_dim))
        self.std_w = UnitLinear(hidden_dim, self.action_dim)
        self.std_bias = nn.Parameter(torch.zeros(self.action_dim))


    def forward(self, x):
        x = x[..., self.policy_observation_indices]
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_norm(x)
        mean = F.linear(x, self.mean_w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.weight, self.std_bias)
        log_std = self.LOG_STD_MIN + (self.LOG_STD_MAX - self.LOG_STD_MIN) * 0.5 * (1.0 + torch.tanh(raw_log_std))
        return mean, torch.exp(log_std)


    def sample_and_log_prob(self, mean, std):
        dist = Normal(mean, std)
        base_sample = dist.rsample()
        action = torch.tanh(base_sample)
        log_prob = dist.log_prob(base_sample)
        tanh_correction = 2.0 * (math.log(2.0) - base_sample - F.softplus(-2.0 * base_sample))
        log_prob = (log_prob - tanh_correction).sum(-1)
        return action, log_prob


    def sample_with_noise(self, mean, std, noise, temperature):
        return torch.tanh(mean + std * noise * temperature)


    def deterministic_action(self, mean):
        return torch.tanh(mean)


    def get_processed_action(self, action):
        clipped_action = torch.clamp(action, -1.0, 1.0)
        return self.env_as_low + 0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low)
