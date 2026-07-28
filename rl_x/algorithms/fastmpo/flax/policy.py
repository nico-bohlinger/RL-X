from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling
from jax import random
from jax._src import dtypes
import flax.linen as nn
from flax.linen.initializers import constant, normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if config.algorithm.policy_network_type == "fastsac":
            policy = FastSACPolicy(env.single_action_space.shape, config.algorithm.policy_init_scale, config.algorithm.policy_min_scale, policy_observation_indices)
        elif config.algorithm.policy_network_type == "fasttd3":
            policy = FastTD3Policy(env.single_action_space.shape, config.algorithm.policy_init_scale, config.algorithm.policy_min_scale, policy_observation_indices)
        elif config.algorithm.policy_network_type == "mpo":
            policy = MPOPolicy(env.single_action_space.shape, config.algorithm.policy_init_scale, config.algorithm.policy_min_scale, policy_observation_indices)

        env_as_scale = env.single_action_space.scale
        env_as_center = env.single_action_space.center
        env_as_low = env.single_action_space.low
        env_as_high = env.single_action_space.high
        max_range = jnp.maximum(jnp.abs(env_as_low - env_as_center), jnp.abs(env_as_high - env_as_center))
        return policy, get_processed_action_function(config.algorithm.action_clipping, config.algorithm.action_rescaling, env_as_low, env_as_high, max_range / env_as_scale)


class FastSACPolicy(nn.Module):
    as_shape: Sequence[int]
    policy_init_scale: float
    policy_min_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        torso = nn.silu(nn.LayerNorm()(nn.Dense(512)(x[..., self.policy_observation_indices])))
        torso = nn.silu(nn.LayerNorm()(nn.Dense(256)(torso)))
        torso = nn.silu(nn.LayerNorm()(nn.Dense(128)(torso)))
        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=constant(0.0), bias_init=constant(0.0))(torso)
        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=constant(0.0), bias_init=constant(0.0))(torso)
        return mean, self.policy_min_scale + jax.nn.softplus(stddev) * self.policy_init_scale / jax.nn.softplus(0.0)


class FastTD3Policy(nn.Module):
    as_shape: Sequence[int]
    policy_init_scale: float
    policy_min_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        torso = nn.relu(nn.Dense(512)(x[..., self.policy_observation_indices]))
        torso = nn.relu(nn.Dense(256)(torso))
        torso = nn.relu(nn.Dense(128)(torso))
        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=normal(0.01), bias_init=constant(0.0))(torso)
        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=constant(0.0), bias_init=constant(0.0))(torso)
        return mean, self.policy_min_scale + jax.nn.softplus(stddev) * self.policy_init_scale / jax.nn.softplus(0.0)


def uniform_scaling(scale, dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        max_value = jnp.sqrt(3 / shape[0]) * scale
        return random.uniform(key, shape, dtype, -max_value, max_value)
    return init


class MPOPolicy(nn.Module):
    as_shape: Sequence[int]
    policy_init_scale: float
    policy_min_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        torso = nn.tanh(nn.LayerNorm()(nn.Dense(512, kernel_init=uniform_scaling(0.333))(x[..., self.policy_observation_indices])))
        torso = nn.elu(nn.Dense(256, kernel_init=uniform_scaling(0.333))(torso))
        torso = nn.elu(nn.Dense(128, kernel_init=uniform_scaling(0.333))(torso))
        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(torso)
        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(torso)
        return mean, self.policy_min_scale + jax.nn.softplus(stddev) * self.policy_init_scale / jax.nn.softplus(0.0)


def get_processed_action_function(action_clipping, action_rescaling, env_as_low, env_as_high, action_scale):
    def get_processed_action(action):
        if action_clipping:
            action = jnp.clip(action, -1, 1)
        if action_rescaling == "normal":
            action = env_as_low + 0.5 * (action + 1.0) * (env_as_high - env_as_low)
        elif action_rescaling == "fastsac":
            action = action * action_scale
        return action
    return jax.jit(get_processed_action)
