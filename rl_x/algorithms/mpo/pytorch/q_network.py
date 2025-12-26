import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_q_network(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
    compile_mode = config.algorithm.compile_mode

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        q_network = torch.compile(QNetwork(env, config.algorithm.nr_atoms, config.algorithm.action_clipping, config.algorithm.nr_hidden_units, device, critic_observation_indices).to(device), mode=compile_mode)
        q_network.forward = torch.compile(q_network.forward, mode=compile_mode)
        return q_network


class QNetwork(nn.Module):
    def __init__(self, env, nr_atoms, action_clipping, nr_hidden_units, device, critic_observation_indices):
        super().__init__()
        self.action_clipping = action_clipping
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        single_as_shape = env.single_action_space.shape
        obs_input_dim = len(critic_observation_indices)
        action_input_dim = np.prod(single_as_shape, dtype=int).item()

        self.critic = nn.Sequential(
            self.uniform_scaling_layer_init(nn.Linear(obs_input_dim + action_input_dim, nr_hidden_units)),
            nn.LayerNorm(nr_hidden_units),
            nn.Tanh(),
            self.uniform_scaling_layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.ELU(),
            self.uniform_scaling_layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.ELU(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_atoms), std=1e-5, variance_scaling=True),
        )


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

    
    def forward(self, x, a):
        if self.action_clipping:
            a = torch.clamp(a, -1.0, 1.0)
        x = x[..., self.critic_observation_indices]
        return self.critic(torch.cat([x, a], dim=1))
