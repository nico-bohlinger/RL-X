from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from jax import random
from jax import core
from jax._src import dtypes
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.init_stddev, config.algorithm.min_stddev, config.algorithm.nr_hidden_units),
                get_processed_action_function(jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)))


def uniform_scaling(scale, dtype = jnp.float_):
    def init(key, shape, dtype=dtype):
        input_size = np.product(shape[:-1])
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        max_val = jnp.sqrt(3 / input_size) * scale
        return random.uniform(key, named_shape, dtype, -1) * max_val
    return init


class Policy(nn.Module):
    as_shape: Sequence[int]
    init_stddev: float
    min_stddev: float
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)

        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(x)

        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(x)
        stddev = self.min_stddev + (jax.nn.softplus(stddev) * self.init_stddev / jax.nn.softplus(0.0))  # Cholesky factor from MPO paper, implemented like in https://github.com/deepmind/acme/blob/master/acme/jax/networks/distributional.py

        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

        return dist


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
