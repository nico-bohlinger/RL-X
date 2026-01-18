from typing import Sequence
import numpy as np
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from jax import random
from jax._src import dtypes
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if config.algorithm.critic_network_type == "fastsac":
            return FastSACVectorCritic(config.algorithm.dual_critic, config.algorithm.nr_atoms, critic_observation_indices)
        elif config.algorithm.critic_network_type == "fasttd3":
            return FastTD3VectorCritic(config.algorithm.dual_critic, config.algorithm.nr_atoms, critic_observation_indices)
        elif config.algorithm.critic_network_type == "mpo":
            return MPOVectorCritic(config.algorithm.dual_critic, config.algorithm.nr_atoms, critic_observation_indices)


class FastSACCritic(nn.Module):
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = x[..., self.critic_observation_indices]
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(768)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dense(384)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dense(192)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = nn.Dense(self.nr_atoms)(x)
        return x


class FastTD3Critic(nn.Module):
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = x[..., self.critic_observation_indices]
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_atoms)(x)
        return x


def uniform_scaling(scale, dtype = jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        fan_in = shape[0]
        max_val = jnp.sqrt(3 / fan_in) * scale
        return random.uniform(key, shape, dtype, -max_val, max_val)
    return init


class MPOCritic(nn.Module):
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = x[..., self.critic_observation_indices]
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(512, kernel_init=uniform_scaling(0.333))(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = nn.Dense(256, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(128, kernel_init=uniform_scaling(0.333))(x)
        x = nn.elu(x)
        x = nn.Dense(self.nr_atoms, kernel_init=variance_scaling(1e-5, "fan_in", "truncated_normal"))(x)
        return x
    

class FastSACVectorCritic(nn.Module):
    dual_critic: bool
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs: np.ndarray, action: np.ndarray):
        vmap_critic = nn.vmap(
            FastSACCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=2 if self.dual_critic else 1,
        )
        q_values = vmap_critic(nr_atoms=self.nr_atoms, critic_observation_indices=self.critic_observation_indices)(obs, action)
        return q_values


class FastTD3VectorCritic(nn.Module):
    dual_critic: bool
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs: np.ndarray, action: np.ndarray):
        vmap_critic = nn.vmap(
            FastTD3Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=2 if self.dual_critic else 1,
        )
        q_values = vmap_critic(nr_atoms=self.nr_atoms, critic_observation_indices=self.critic_observation_indices)(obs, action)
        return q_values


class MPOVectorCritic(nn.Module):
    dual_critic: bool
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs: np.ndarray, action: np.ndarray):
        vmap_critic = nn.vmap(
            MPOCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=2 if self.dual_critic else 1,
        )
        q_values = vmap_critic(nr_atoms=self.nr_atoms, critic_observation_indices=self.critic_observation_indices)(obs, action)
        return q_values
