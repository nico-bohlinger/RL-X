import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_q_network(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        q_network = torch.compile(QNetwork(env, config.algorithm.nr_atoms, critic_observation_indices).to(device), mode=compile_mode)
        q_network.forward = torch.compile(q_network.forward, mode=compile_mode)
        return q_network


class QNetwork(nn.Module):
    def __init__(self, env, nr_atoms, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = critic_observation_indices

        nr_observations = len(critic_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()

        self.critic = nn.Sequential(
            nn.Linear(nr_observations + nr_actions, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, nr_atoms),
        )

    
    def forward(self, x, a):
        x = x[..., self.critic_observation_indices]
        return self.critic(torch.cat([x, a], dim=1))
