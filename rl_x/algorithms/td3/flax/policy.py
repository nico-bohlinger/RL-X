from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.nr_hidden_units),
                get_processed_action_function(jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)))



class Policy(nn.Module):
    as_shape: Sequence[int]
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(np.prod(self.as_shape).item())(x)
        x = nn.tanh(x)

        return x


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
