import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Policy(env, config.algorithm.log_std_min, config.algorithm.log_std_max, config.algorithm.nr_hidden_units, device)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")
    

class Policy(nn.Module):
    def __init__(self, env, log_std_min, log_std_max, nr_hidden_units, device):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.env_as_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
        self.env_as_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        self.torso = nn.Sequential(
            nn.Linear(np.prod(single_os_shape), nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
        )
        self.mean = nn.Linear(nr_hidden_units, single_as_shape)
        self.log_std = nn.Linear(nr_hidden_units, single_as_shape)


    def get_action(self, x):
        latent = self.torso(x)
        mean = self.mean(latent)
        log_std = self.log_std(latent)

        # Only CleanRL does it, not SBX or Jaxrl. They just clip log_std to min and max
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # Reparameterization trick
        action_tanh = torch.tanh(action)

        log_prob = normal.log_prob(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        scaled_action = self.env_as_low + (0.5 * (action_tanh + 1.0) * (self.env_as_high - self.env_as_low))

        return action_tanh, scaled_action, log_prob


    def get_deterministic_action(self, x):
        latent = self.torso(x)
        mean = self.mean(latent)
        action_tanh = torch.tanh(mean)
        scaled_action = self.env_as_low + (0.5 * (action_tanh + 1.0) * (self.env_as_high - self.env_as_low))
        return scaled_action
    

    def get_random_actions(self, env, nr_envs):
            scaled_action = np.array([env.action_space.sample() for _ in range(nr_envs)])
            action = (scaled_action - self.env_as_low.cpu().numpy()) / (self.env_as_high.cpu().numpy() - self.env_as_low.cpu().numpy()) * 2.0 - 1.0
            return action, scaled_action
