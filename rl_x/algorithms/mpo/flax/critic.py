from typing import Sequence
import numpy as np
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from jax import random
from jax import core
from jax._src import dtypes
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(config.algorithm.nr_atoms, config.algorithm.action_clipping, config.algorithm.nr_hidden_units, critic_observation_indices)


def uniform_scaling(scale, dtype = jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        fan_in = shape[0]
        max_val = jnp.sqrt(3 / fan_in) * scale
        return random.uniform(key, shape, dtype, -max_val, max_val)
    return init


class Critic(nn.Module):
    nr_atoms: int
    action_clipping: bool
    nr_hidden_units: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = x[..., self.critic_observation_indices]
        if self.action_clipping:
            a = jnp.clip(a, -1, 1)
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(self.nr_atoms, kernel_init=variance_scaling(1e-5, "fan_in", "truncated_normal"))(x)
        return x
