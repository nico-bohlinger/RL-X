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

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorCritic(config.algorithm.nr_atoms_per_net, config.algorithm.nr_hidden_units, config.algorithm.ensemble_size)


def uniform_scaling(scale, dtype = jnp.float_):
    def init(key, shape, dtype=dtype):
        input_size = np.product(shape[:-1])
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        max_val = jnp.sqrt(3 / input_size) * scale
        return random.uniform(key, named_shape, dtype, -1) * max_val
    return init


class Critic(nn.Module):
    nr_atoms: int
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        a = jnp.clip(a, -1, 1)
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(self.nr_hidden_units, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(self.nr_atoms, kernel_init=variance_scaling(0.01, "fan_in", "truncated_normal"))(x)
        return x
    

class VectorCritic(nn.Module):
    nr_atoms_per_net: int
    nr_hidden_units: int
    nr_critics: int

    @nn.compact
    def __call__(self, obs: np.ndarray, action: np.ndarray):
        # Reference:
        # - https://github.com/araffin/sbx/blob/f31288d2701b39dd98c921f55e13ca3530868e9f/sbx/sac/policies.py
        # - https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/networks/critic_net.py

        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )
        q_values = vmap_critic(nr_atoms=self.nr_atoms_per_net, nr_hidden_units=self.nr_hidden_units)(obs, action)
        return q_values
