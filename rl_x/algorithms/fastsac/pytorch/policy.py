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
        policy = torch.compile(Policy(env, config.algorithm.log_std_min, config.algorithm.log_std_max, device, policy_observation_indices).to(device), mode=compile_mode)
        policy.forward = torch.compile(policy.forward, mode=compile_mode)
        policy.get_action_and_log_prob = torch.compile(policy.get_action_and_log_prob, mode=compile_mode)
        policy.get_action = torch.compile(policy.get_action, mode=compile_mode)
        return policy


class Policy(nn.Module):
    def __init__(self, env, log_std_min, log_std_max, device, policy_observation_indices):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)

        env_as_scale = torch.tensor(env.single_action_space.scale, dtype=torch.float32, device=device)
        env_as_center = torch.tensor(env.single_action_space.center, dtype=torch.float32, device=device)
        env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        range_to_lower = torch.abs(env_as_low - env_as_center)
        range_to_upper = torch.abs(env_as_high - env_as_center)
        max_range = torch.maximum(range_to_lower, range_to_upper)
        self.action_scale = max_range / env_as_scale

        nr_observations = len(policy_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()

        self.torso = nn.Sequential(
            nn.Linear(nr_observations, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.mean = self.layer_init(nn.Linear(128, nr_actions), weight_const=0.0, bias_const=0.0)
        self.log_std = self.layer_init(nn.Linear(128, nr_actions), weight_const=0.0, bias_const=0.0)


    def layer_init(self, layer, weight_const, bias_const):
        nn.init.constant_(layer.weight, weight_const)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def forward(self, x):
        x = x[..., self.policy_observation_indices]
        latent = self.torso(x)
        mean = self.mean(latent)
        log_std = self.log_std(latent)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mean, log_std


    def get_action_and_log_prob(self, x):
        mean, log_std = self.forward(x)

        std = log_std.exp()

        dist = Normal(mean, std)
        action_raw = dist.rsample()
        action_tanh = torch.tanh(action_raw)
        action_scaled = action_tanh * self.action_scale

        log_prob = dist.log_prob(action_raw)
        log_prob -= torch.log((1 - action_tanh.pow(2)) + 1e-6)
        log_prob -= torch.log(self.action_scale + 1e-6)
        log_prob = log_prob.sum(1)

        return action_scaled, log_prob
    

    def get_action(self, x, deterministic=False):
        mean, log_std = self.forward(x)

        if deterministic:
            action_raw = mean
            action_tanh = torch.tanh(action_raw)
            action_scaled = action_tanh * self.action_scale
            return action_scaled
        
        std = log_std.exp()

        dist = Normal(mean, std)
        action_raw = dist.rsample()
        action_tanh = torch.tanh(action_raw)
        action_scaled = action_tanh * self.action_scale

        return action_scaled
