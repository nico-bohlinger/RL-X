from typing import Sequence
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(critic_observation_indices)


class Critic(nn.Module):
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.critic_observation_indices]
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.LayerNorm()(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic)
        return critic
