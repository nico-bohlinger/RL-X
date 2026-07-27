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
        policy = Policy(config, env, device, policy_observation_indices).to(device)
        policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
        policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
        return policy


class Policy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = config.algorithm.action_clipping_and_rescaling
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()
        hidden_dim = config.algorithm.nr_hidden_units
        self.policy_mean = nn.Sequential(nn.Linear(len(policy_observation_indices), hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))
        nn.init.orthogonal_(self.policy_mean[0].weight, np.sqrt(2.0))
        nn.init.zeros_(self.policy_mean[0].bias)
        nn.init.orthogonal_(self.policy_mean[2].weight, np.sqrt(2.0))
        nn.init.zeros_(self.policy_mean[2].bias)
        nn.init.orthogonal_(self.policy_mean[4].weight, 0.01)
        nn.init.zeros_(self.policy_mean[4].bias)
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), np.log(config.algorithm.std_dev).item()))


    def forward(self, x):
        mean = self.policy_mean(x[..., self.policy_observation_indices])
        return mean, self.policy_logstd.expand_as(mean)


    def get_action_logprob(self, x):
        mean, log_std = self(x)
        noise = torch.randn_like(mean)
        action = mean + torch.exp(log_std) * noise
        log_prob = (-0.5 * noise ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std).sum(dim=-1)
        return action, self.process_action(action), log_prob


    def get_logprob(self, x, action):
        mean, log_std = self(x)
        return (-0.5 * ((action - mean) / torch.exp(log_std)) ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std).sum(dim=-1)


    def process_action(self, action):
        if self.action_clipping_and_rescaling:
            action = torch.clamp(action, -1.0, 1.0)
            return self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
        return action


    def get_deterministic_action(self, x):
        mean, _ = self(x)
        return self.process_action(mean)
