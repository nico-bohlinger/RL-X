import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        critic = torch.compile(Critic(env, config.algorithm.nr_hidden_units, device, critic_observation_indices).to(device), mode=compile_mode)
        critic.get_value = torch.compile(critic.get_value, mode=compile_mode)
        return critic


class Critic(nn.Module):
    def __init__(self, env, nr_hidden_units, device, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        obs_input_dim = len(critic_observation_indices)

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(obs_input_dim, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, 1), std=1.0),
        )


    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_value(self, x):
        x = x[..., self.critic_observation_indices]
        return self.critic(x)
