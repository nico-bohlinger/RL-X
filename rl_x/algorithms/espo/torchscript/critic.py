import numpy as np
import torch
import torch.nn as nn

from rl_x.algorithms.ppo.torchscript.normal_distribution import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(env, config.algorithm.nr_hidden_units, config.algorithm.critic_coef)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class Critic(nn.Module):
    def __init__(self, env, nr_hidden_units, critic_coef: float):
        super().__init__()
        self.critic_coef = critic_coef
        single_os_shape = env.observation_space.shape

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape, dtype=int).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, 1), std=1.0),
        )


    def layer_init(self, layer, std=np.sqrt(2).item(), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    @torch.jit.export
    def get_value(self, x):
        return self.critic(x)
    

    @torch.jit.export
    def loss(self, states, returns):
        new_value = self.get_value(states).reshape(-1)
        v_loss = (0.5 * (new_value - returns) ** 2).mean()

        return self.critic_coef * v_loss, v_loss
