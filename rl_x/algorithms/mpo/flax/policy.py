from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.get_single_action_space_shape(), config.algorithm.init_stddev, config.algorithm.min_stddev, config.algorithm.nr_hidden_units),
                get_processed_action_function(jnp.array(env.action_space.low), jnp.array(env.action_space.high)))
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")



class Policy(nn.Module):
    as_shape: Sequence[int]
    init_stddev: float
    min_stddev: float
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)

        mean = nn.Dense(np.prod(self.as_shape).item())(x)

        stddev = nn.Dense(np.prod(self.as_shape).item())(x)
        stddev = self.min_stddev + (jax.nn.softplus(stddev) * self.init_stddev / jax.nn.softplus(0.0))  # Cholesky factor from MPO paper, implemented like in https://github.com/deepmind/acme/blob/master/acme/jax/networks/distributional.py

        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

        return dist


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
