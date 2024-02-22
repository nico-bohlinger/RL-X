import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(config.algorithm.nr_hidden_units)


class Critic(nn.Module):
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
