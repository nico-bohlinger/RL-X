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
        policy = torch.compile(Policy(config, env, device, policy_observation_indices).to(device), mode=config.algorithm.compile_mode)
        policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
        policy.get_logprob_entropy = torch.compile(policy.get_logprob_entropy, mode=config.algorithm.compile_mode)
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
        dimensions = [len(policy_observation_indices), 256, 256, 128, 128, 64, 64, action_dim]
        layers = []
        for layer_index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layer = nn.Linear(input_dim, output_dim)
            nn.init.orthogonal_(layer.weight, 0.01 if layer_index == len(dimensions) - 2 else np.sqrt(2.0))
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.Tanh()])
        self.policy_mean = nn.Sequential(*layers[:-1])
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), np.log(config.algorithm.std_dev).item()))


    def forward(self, x):
        mean = self.policy_mean(x[..., self.policy_observation_indices])
        return mean, self.policy_logstd.expand_as(mean)


    def get_action_logprob(self, x):
        mean, log_std = self(x)
        noise = torch.randn_like(mean)
        action = mean + torch.exp(log_std) * noise
        log_prob = (-0.5 * noise ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std).sum(dim=-1)
        processed_action = torch.clamp(action, self.env_as_low, self.env_as_high) if self.action_clipping_and_rescaling else action
        return action, processed_action, log_prob


    def get_logprob_entropy(self, x, action):
        mean, log_std = self(x)
        log_prob = (-0.5 * ((action - mean) / torch.exp(log_std)) ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std).sum(dim=-1)
        entropy = (log_std + 0.5 * np.log(2.0 * np.pi * np.e)).sum(dim=-1)
        return log_prob, entropy


    def get_deterministic_action(self, x):
        mean, _ = self(x)
        return torch.clamp(mean, self.env_as_low, self.env_as_high) if self.action_clipping_and_rescaling else mean
