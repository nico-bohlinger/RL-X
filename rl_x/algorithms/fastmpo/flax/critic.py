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
        if config.algorithm.critic_network_type == "fasttd3":
            return FastTD3VectorCritic(config.algorithm.dual_critic, config.algorithm.nr_atoms, critic_observation_indices)
        if config.algorithm.critic_network_type == "mpo":
            return MPOVectorCritic(config.algorithm.dual_critic, config.algorithm.nr_atoms, critic_observation_indices)


class FastSACCritic(nn.Module):
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x, action):
        x = jnp.concatenate([x[..., self.critic_observation_indices], action], axis=-1)
        x = nn.silu(nn.LayerNorm()(nn.Dense(768)(x)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(384)(x)))
        x = nn.silu(nn.LayerNorm()(nn.Dense(192)(x)))
        return nn.Dense(self.nr_atoms)(x)


class FastTD3Critic(nn.Module):
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x, action):
        x = jnp.concatenate([x[..., self.critic_observation_indices], action], axis=-1)
        x = nn.relu(nn.Dense(1024)(x))
        x = nn.relu(nn.Dense(512)(x))
        x = nn.relu(nn.Dense(256)(x))
        return nn.Dense(self.nr_atoms)(x)


def uniform_scaling(scale, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        max_value = jnp.sqrt(3 / shape[0]) * scale
        return random.uniform(key, shape, dtype, -max_value, max_value)
    return init


class MPOCritic(nn.Module):
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x, action):
        x = jnp.concatenate([x[..., self.critic_observation_indices], action], axis=-1)
        x = nn.tanh(nn.LayerNorm()(nn.Dense(512, kernel_init=uniform_scaling(0.333))(x)))
        x = nn.elu(nn.Dense(256, kernel_init=uniform_scaling(0.333))(x))
        x = nn.elu(nn.Dense(128, kernel_init=uniform_scaling(0.333))(x))
        return nn.Dense(self.nr_atoms, kernel_init=variance_scaling(1e-5, "fan_in", "truncated_normal"))(x)


class FastSACVectorCritic(nn.Module):
    dual_critic: bool
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action):
        critic = nn.vmap(FastSACCritic, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0, axis_size=2 if self.dual_critic else 1)
        return critic(nr_atoms=self.nr_atoms, critic_observation_indices=self.critic_observation_indices)(observation, action)


class FastTD3VectorCritic(nn.Module):
    dual_critic: bool
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action):
        critic = nn.vmap(FastTD3Critic, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0, axis_size=2 if self.dual_critic else 1)
        return critic(nr_atoms=self.nr_atoms, critic_observation_indices=self.critic_observation_indices)(observation, action)


class MPOVectorCritic(nn.Module):
    dual_critic: bool
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action):
        critic = nn.vmap(MPOCritic, variable_axes={"params": 0}, split_rngs={"params": True}, in_axes=None, out_axes=0, axis_size=2 if self.dual_critic else 1)
        return critic(nr_atoms=self.nr_atoms, critic_observation_indices=self.critic_observation_indices)(observation, action)
