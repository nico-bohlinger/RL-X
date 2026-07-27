import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        critic = Critic(device, critic_observation_indices).to(device)

    critic.get_value = torch.compile(critic.get_value, mode=config.algorithm.compile_mode)
    return critic


class Critic(nn.Module):
    def __init__(self, device, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        self.critic_dense1 = self.layer_init(nn.Linear(len(critic_observation_indices), 512))
        self.critic_ln1 = nn.LayerNorm(512, eps=1e-6)
        self.critic_dense2 = self.layer_init(nn.Linear(512, 256))
        self.critic_dense3 = self.layer_init(nn.Linear(256, 128))
        self.critic_head = self.layer_init(nn.Linear(128, 1), std=1.0)


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def get_value(self, x):
        x = x[..., self.critic_observation_indices]
        x = torch.nn.functional.elu(self.critic_ln1(self.critic_dense1(x)))
        x = torch.nn.functional.elu(self.critic_dense2(x))
        x = torch.nn.functional.elu(self.critic_dense3(x))
        return self.critic_head(x)
