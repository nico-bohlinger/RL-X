from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    env_as_scale = env.single_action_space.scale
    env_as_center = env.single_action_space.center
    env_as_low = env.single_action_space.low
    env_as_high = env.single_action_space.high
    range_to_lower = jnp.abs(env_as_low - env_as_center)
    range_to_upper = jnp.abs(env_as_high - env_as_center)
    max_range = jnp.maximum(range_to_lower, range_to_upper)
    action_scale = max_range / env_as_scale

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Policy(env.single_action_space.shape, config.algorithm.log_std_min, config.algorithm.log_std_max, action_scale, policy_observation_indices)


class Policy(nn.Module):
    as_shape: Sequence[int]
    log_std_min: float
    log_std_max: float
    action_scale: jnp.ndarray
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
        log_std = nn.Dense(np.prod(self.as_shape).item(), kernel_init=constant(0.0), bias_init=constant(0.0))(torso)
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std
    

    def get_action(self, mean, log_std, key):
        std = jnp.exp(log_std)
        action_raw = mean + std * jax.random.normal(key, shape=mean.shape)
        action_tanh = jnp.tanh(action_raw)
        action_scaled = action_tanh * self.action_scale
        return action_scaled
    

    def get_deterministic_action(self, mean):
        action_tanh = jnp.tanh(mean)
        action_scaled = action_tanh * self.action_scale
        return action_scaled
    

    def get_action_and_log_prob(self, mean, log_std, key):
        std = jnp.exp(log_std)
        action_raw = mean + std * jax.random.normal(key, shape=mean.shape)
        action_tanh = jnp.tanh(action_raw)
        action_scaled = action_tanh * self.action_scale

        log_prob = -0.5 * ((action_raw - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
        log_prob -= jnp.log(1.0 - action_tanh ** 2 + 1e-6)
        log_prob -= jnp.log(self.action_scale + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1)

        return action_scaled, log_prob
