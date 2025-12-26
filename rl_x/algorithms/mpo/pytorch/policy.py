import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = torch.compile(Policy(env, config.algorithm.policy_init_scale, config.algorithm.policy_min_scale, config.algorithm.action_clipping, config.algorithm.action_rescaling, config.algorithm.nr_hidden_units, device, policy_observation_indices).to(device), mode=compile_mode)
        policy.forward = torch.compile(policy.forward, mode=compile_mode)
        policy.get_action = torch.compile(policy.get_action, mode=compile_mode)
        policy.get_distribution = torch.compile(policy.get_distribution, mode=compile_mode)
        policy.sample_action = torch.compile(policy.sample_action, mode=compile_mode)
        policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=compile_mode)
        return policy
    

class Policy(nn.Module):
    def __init__(self, env, policy_init_scale, policy_min_scale, action_clipping, action_rescaling, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.policy_init_scale = policy_init_scale
        self.policy_min_scale = policy_min_scale
        self.action_clipping = action_clipping
        self.action_rescaling = action_rescaling
        self.device = device
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32).to(device)
        single_as_shape = env.single_action_space.shape
        obs_input_dim = len(policy_observation_indices)

        self.torso = nn.Sequential(
            self.uniform_scaling_layer_init(nn.Linear(obs_input_dim, nr_hidden_units)),
            nn.LayerNorm(nr_hidden_units),
            nn.Tanh(),
            self.uniform_scaling_layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.ELU(),
            self.uniform_scaling_layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.ELU(),
        )
        self.mean = self.layer_init(nn.Linear(nr_hidden_units, np.prod(single_as_shape, dtype=int).item()), std=1e-4, variance_scaling=True)
        self.std = self.layer_init(nn.Linear(nr_hidden_units, np.prod(single_as_shape, dtype=int).item()), std=1e-4, variance_scaling=True)
        
        self.softplus0 = float(F.softplus(torch.zeros(1)).item())


    def uniform_scaling_layer_init(self, layer, bias_const=0.0, scale=0.333):
        max_val = torch.sqrt(torch.as_tensor(3.0) / torch.as_tensor(layer.weight.shape[1])) * scale
        torch.nn.init.uniform_(layer.weight, a=-float(max_val), b=float(max_val))
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0, variance_scaling=False):
        if variance_scaling:
            std = torch.sqrt(torch.as_tensor(std) / torch.as_tensor(layer.weight.shape[1]))
            distribution_stddev = torch.as_tensor(0.87962566103423978)
            std = std / distribution_stddev
        torch.nn.init.trunc_normal_(layer.weight, std=float(std))
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_action(self, x):
        x = x[..., self.policy_observation_indices]
        latent = self.torso(x)
        mean = self.mean(latent)
        std = self.std(latent)
        std = self.policy_min_scale + (F.softplus(std) * self.policy_init_scale / self.softplus0)
        return mean, std


    def get_distribution(self, x):
        mean, std = self.get_action(x)
        return Independent(Normal(mean, std), 1)


    def scale_to_env(self, action):
        processed_action = action
        if self.action_clipping:
            processed_action = torch.clamp(processed_action, -1.0, 1.0)
        if self.action_rescaling:
            processed_action = self.env_as_low + (0.5 * (processed_action + 1.0) * (self.env_as_high - self.env_as_low))
        return processed_action


    def sample_action(self, x):
        dist = self.get_distribution(x)
        a = dist.rsample()
        return a, self.scale_to_env(a)


    def get_deterministic_action(self, x):
        mean, _ = self.get_action(x)
        return self.scale_to_env(mean)
