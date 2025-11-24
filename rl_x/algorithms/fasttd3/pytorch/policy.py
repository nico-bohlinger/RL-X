import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = torch.compile(Policy(env, config.algorithm.action_clipping_and_rescaling, device, policy_observation_indices).to(device), mode=compile_mode)
        policy.forward = torch.compile(policy.forward, mode=compile_mode)
        policy.get_action = torch.compile(policy.get_action, mode=compile_mode)
        return policy


class Policy(nn.Module):
    def __init__(self, env, action_clipping_and_rescaling, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.policy_observation_indices = policy_observation_indices
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)

        nr_observations = len(policy_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()

        self.policy = nn.Sequential(
            nn.Linear(nr_observations, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, nr_actions),
            nn.Tanh()
        )


    def forward(self, x):
        x = x[..., self.policy_observation_indices]
        action = self.policy(x)
        return action


    def get_action(self, x, noise_scales):
        with torch.no_grad():
            action = self.forward(x)
            if noise_scales is not None:
                noise = torch.randn_like(action) * noise_scales
                action = action + noise
            processed_action = action
            if self.action_clipping_and_rescaling:
                clipped_action = torch.clamp(action, -1.0, 1.0)
                processed_action = self.env_as_low + 0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low)
            return action, processed_action
