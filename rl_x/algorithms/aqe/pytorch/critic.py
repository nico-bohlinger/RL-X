import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.aqe.pytorch.q_network import VectorQNetwork


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(config, env, device)


class Critic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
        self.q = torch.compile(VectorQNetwork(config, env, device, critic_observation_indices).to(device), mode=config.algorithm.compile_mode)
        self.q_target = torch.compile(VectorQNetwork(config, env, device, critic_observation_indices).to(device), mode=config.algorithm.compile_mode)
        self.q_target.load_state_dict(self.q.state_dict())
