import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        critic = Critic(config, device, critic_observation_indices).to(device)
        critic.get_value = torch.compile(critic.get_value, mode=config.algorithm.compile_mode)
        return critic


class Critic(nn.Module):
    def __init__(self, config, device, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        hidden_dim = config.algorithm.nr_hidden_units
        self.critic = nn.Sequential(nn.Linear(len(critic_observation_indices), hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        nn.init.orthogonal_(self.critic[0].weight, np.sqrt(2.0))
        nn.init.zeros_(self.critic[0].bias)
        nn.init.orthogonal_(self.critic[2].weight, np.sqrt(2.0))
        nn.init.zeros_(self.critic[2].bias)
        nn.init.orthogonal_(self.critic[4].weight, 1.0)
        nn.init.zeros_(self.critic[4].bias)


    def get_value(self, x):
        return self.critic(x[..., self.critic_observation_indices])
