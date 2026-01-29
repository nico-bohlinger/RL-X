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
        range_to_lower = jnp.abs(env_as_low - env_as_center)
        range_to_upper = jnp.abs(env_as_high - env_as_center)
        max_range = jnp.maximum(range_to_lower, range_to_upper)
        action_scale = max_range / env_as_scale

        fn = get_processed_action_function(config.algorithm.action_clipping, config.algorithm.action_rescaling, env_as_low, env_as_high, action_scale)

        return (policy, fn)


class FastSACPolicy(nn.Module):
    as_shape: Sequence[int]
    policy_init_scale: float
    policy_min_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        torso = nn.Dense(512)(x)
        torso = nn.LayerNorm()(torso)
        torso = nn.silu(torso)
        torso = nn.Dense(256)(torso)
        torso = nn.LayerNorm()(torso)
        torso = nn.silu(torso)
        torso = nn.Dense(128)(torso)
        torso = nn.LayerNorm()(torso)
        torso = nn.silu(torso)

        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=constant(0.0), bias_init=constant(0.0))(torso)

        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=constant(0.0), bias_init=constant(0.0))(torso)
        stddev = self.policy_min_scale + (jax.nn.softplus(stddev) * self.policy_init_scale / jax.nn.softplus(0.0))

        return mean, stddev


class FastTD3Policy(nn.Module):
    as_shape: Sequence[int]
    policy_init_scale: float
    policy_min_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        torso = nn.Dense(512)(x)
        torso = nn.relu(torso)
        torso = nn.Dense(256)(torso)
        torso = nn.relu(torso)
        torso = nn.Dense(128)(torso)
        torso = nn.relu(torso)

        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=normal(0.01), bias_init=constant(0.0))(torso)

        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=constant(0.0), bias_init=constant(0.0))(torso)
        stddev = self.policy_min_scale + (jax.nn.softplus(stddev) * self.policy_init_scale / jax.nn.softplus(0.0))

        return mean, stddev


def uniform_scaling(scale, dtype = jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        fan_in = shape[0]
        max_val = jnp.sqrt(3 / fan_in) * scale
        return random.uniform(key, shape, dtype, -max_val, max_val)
    return init


class MPOPolicy(nn.Module):
    as_shape: Sequence[int]
    policy_init_scale: float
    policy_min_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        torso = x[..., self.policy_observation_indices]
        torso = nn.Dense(512, kernel_init=uniform_scaling(0.333))(torso)
        torso = nn.LayerNorm()(torso)
        torso = nn.tanh(torso)
        torso = nn.Dense(256, kernel_init=uniform_scaling(0.333))(torso)
        torso = nn.elu(torso)
        torso = nn.Dense(128, kernel_init=uniform_scaling(0.333))(torso)
        torso = nn.elu(torso)

        mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(torso)

        stddev = nn.Dense(np.prod(self.as_shape).item(), kernel_init=variance_scaling(1e-4, "fan_in", "truncated_normal"))(torso)
        stddev = self.policy_min_scale + (jax.nn.softplus(stddev) * self.policy_init_scale / jax.nn.softplus(0.0))

        return mean, stddev


def get_processed_action_function(action_clipping, action_rescaling, env_as_low, env_as_high, action_scale):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        if action_clipping:
            action = jnp.clip(action, -1, 1)
        if action_rescaling == "none":
            pass
        elif action_rescaling == "normal":
            action = env_as_low + (0.5 * (action + 1.0) * (env_as_high - env_as_low))
        elif action_rescaling == "fastsac":
            action = action * action_scale
        return action
    return jax.jit(get_clipped_and_scaled_action)
