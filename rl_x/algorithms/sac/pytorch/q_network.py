import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_q_network(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        q_network = torch.compile(QNetwork(env, config.algorithm.nr_hidden_units, device, critic_observation_indices).to(device), mode=compile_mode)
        q_network.forward = torch.compile(q_network.forward, mode=compile_mode)
        return q_network


class QNetwork(nn.Module):
    def __init__(self, env, nr_hidden_units, device, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        single_as_shape = env.single_action_space.shape
        obs_input_dim = len(critic_observation_indices)
        action_input_dim = np.prod(single_as_shape, dtype=int).item()

        self.critic = nn.Sequential(
            nn.Linear(obs_input_dim + action_input_dim, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, 1),
        )

    
    def forward(self, x, a):
        x = x[..., self.critic_observation_indices]
        return self.critic(torch.cat([x, a], dim=1))
