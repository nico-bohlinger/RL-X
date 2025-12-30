from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from jax import random
from jax import core
from jax._src import dtypes
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.policy_init_scale, config.algorithm.policy_min_scale, config.algorithm.nr_hidden_units, policy_observation_indices),
                get_processed_action_function(config.algorithm.action_clipping, config.algorithm.action_rescaling, jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)))


def uniform_scaling(scale, dtype = jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        fan_in = shape[0]
        max_val = jnp.sqrt(3 / fan_in) * scale
        return random.uniform(key, shape, dtype, -max_val, max_val)
    return init


class Policy(nn.Module):
    as_shape: Sequence[int]
    policy_init_scale: float
    policy_min_scale: float
    nr_hidden_units: int
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)

        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(x)

        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(x)
        stddev = self.policy_min_scale + (jax.nn.softplus(stddev) * self.policy_init_scale / jax.nn.softplus(0.0))  # Cholesky factor from MPO paper, implemented like in https://github.com/deepmind/acme/blob/master/acme/jax/networks/distributional.py

        return mean, stddev


def get_processed_action_function(action_clipping, action_rescaling, env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        if action_clipping:
            action = jnp.clip(action, -1, 1)
        if action_rescaling:
            action = env_as_low + (0.5 * (action + 1.0) * (env_as_high - env_as_low))
        return action
    return jax.jit(get_clipped_and_scaled_action)
