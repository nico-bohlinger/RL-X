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
        return torch.compile(Policy(env, config.algorithm.nr_hidden_units, device, policy_observation_indices).to(device), mode=config.algorithm.compile_mode)


class Policy(nn.Module):
    def __init__(self, env, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        self.network = nn.Sequential(
            nn.Linear(len(policy_observation_indices), nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, np.prod(env.single_action_space.shape, dtype=int).item()),
            nn.Tanh(),
        )


    def forward(self, x):
        return self.network(x[..., self.policy_observation_indices])


    def get_processed_action(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        return self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
