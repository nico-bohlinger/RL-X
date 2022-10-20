from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_actor(config, env):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Actor(env.get_single_action_space_shape(), config.algorithm.std_dev, config.algorithm.nr_hidden_units),
                get_processed_action_function(jnp.array(env.action_space.low), jnp.array(env.action_space.high)))
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class Actor(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.as_shape, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logstd = self.param("actor_logstd", constant(jnp.log(self.std_dev)), (1, self.as_shape))
        return actor_mean, actor_logstd


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
