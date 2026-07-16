import math
import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Policy(
            env=env,
            hidden_dim=config.algorithm.policy_hidden_dim,
            min_std=config.algorithm.policy_min_std,
            init_entropy_coefficient=config.algorithm.init_entropy_coefficient,
            init_kl_coefficient=config.algorithm.init_kl_coefficient,
            policy_observation_indices=policy_observation_indices,
            device=device,
        ).to(device)


class Policy(nn.Module):
    def __init__(self, env, hidden_dim, min_std, init_entropy_coefficient, init_kl_coefficient, policy_observation_indices, device):
        super().__init__()
        self.min_std = min_std
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        env_as_low = torch.as_tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        env_as_high = torch.as_tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        if hasattr(env.single_action_space, "center") and hasattr(env.single_action_space, "scale"):
            env_as_center = torch.as_tensor(env.single_action_space.center, dtype=torch.float32, device=device)
            env_as_scale = torch.as_tensor(env.single_action_space.scale, dtype=torch.float32, device=device)
            env_as_low = (env_as_low - env_as_center) / env_as_scale
            env_as_high = (env_as_high - env_as_center) / env_as_scale
        self.register_buffer("env_as_low", env_as_low)
        self.register_buffer("env_as_high", env_as_high)
        nr_observations = len(policy_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()

        self.torso = nn.Sequential(
            nn.Linear(nr_observations, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, nr_actions * 2)

        self.log_entropy_coefficient = nn.Parameter(torch.full((1,), math.log(init_entropy_coefficient)))
        self.log_kl_coefficient = nn.Parameter(torch.full((1,), math.log(init_kl_coefficient)))


    def forward(self, x):
        x = x[..., self.policy_observation_indices]
        out = self.head(self.torso(x))
        loc, log_std = torch.chunk(out, 2, dim=-1)
        return loc, log_std


    def sample_and_log_prob(self, loc, log_std):
        std = log_std.exp() + self.min_std
        noise = torch.randn_like(loc)
        base_sample = loc + std * noise
        action = torch.tanh(base_sample)
        gaussian_log_prob = -0.5 * noise.pow(2) - 0.5 * math.log(2.0 * math.pi) - torch.log(std)
        tanh_correction = 2.0 * (math.log(2.0) - base_sample - torch.nn.functional.softplus(-2.0 * base_sample))
        log_prob = (gaussian_log_prob - tanh_correction).sum(-1)
        return action, log_prob


    def log_prob(self, loc, log_std, action):
        std = log_std.exp() + self.min_std
        eps = 1e-6
        clipped_action = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        base_sample = torch.atanh(clipped_action)
        gaussian_log_prob = -0.5 * ((base_sample - loc) / std).pow(2) - 0.5 * math.log(2.0 * math.pi) - torch.log(std)
        tanh_correction = 2.0 * (math.log(2.0) - base_sample - torch.nn.functional.softplus(-2.0 * base_sample))
        return (gaussian_log_prob - tanh_correction).sum(-1)


    def deterministic_action(self, loc):
        return torch.tanh(loc)


    def get_processed_action(self, action):
        clipped_action = torch.clamp(action, -1.0, 1.0)
        return self.env_as_low + 0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low)
