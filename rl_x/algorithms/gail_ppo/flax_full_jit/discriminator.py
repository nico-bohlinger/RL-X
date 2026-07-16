import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from collections import deque


def get_discriminator(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Discriminator(config.algorithm.handle_absorbing_states)


class Discriminator(nn.Module):
    handle_absorbing_states: bool

    @nn.compact
    def __call__(self, x, y, x_n, absorbing, shaping=None):
        """
        D(s,a)

        Args:
            x : state
            y: action 
            x_n: next state (not used)
            absorbing: bool for whether the next state is absorbing
        """
        if self.handle_absorbing_states:
            x = jnp.concatenate([x.flatten(), absorbing.flatten(), y.flatten()])
        else:
            x = jnp.concatenate([x.flatten(), y.flatten()])
        discriminator = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        discriminator = nn.tanh(discriminator)
        discriminator = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(discriminator)
        discriminator = nn.tanh(discriminator)
        discriminator = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(discriminator)
        return discriminator
