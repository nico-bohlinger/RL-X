import math
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
    env_as_low = jnp.array(env.single_action_space.low)
    env_as_high = jnp.array(env.single_action_space.high)
    if hasattr(env.single_action_space, "center") and hasattr(env.single_action_space, "scale"):
        env_as_low = (env_as_low - jnp.array(env.single_action_space.center)) / jnp.array(env.single_action_space.scale)
        env_as_high = (env_as_high - jnp.array(env.single_action_space.center)) / jnp.array(env.single_action_space.scale)

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (
            Policy(
                as_shape=env.single_action_space.shape,
                hidden_dim=config.algorithm.policy_hidden_dim,
                min_std=config.algorithm.policy_min_std,
                init_entropy_coefficient=config.algorithm.init_entropy_coefficient,
                init_kl_coefficient=config.algorithm.init_kl_coefficient,
                policy_observation_indices=policy_observation_indices,
            ),
            get_processed_action_function(env_as_low, env_as_high),
        )


class Policy(nn.Module):
    as_shape: Sequence[int]
    hidden_dim: int
    min_std: float
    init_entropy_coefficient: float
    init_kl_coefficient: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        action_dim = np.prod(self.as_shape).item()

        h = nn.Dense(self.hidden_dim)(x)
        h = nn.RMSNorm()(h)
        h = nn.swish(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.RMSNorm()(h)
        h = nn.swish(h)
        out = nn.Dense(action_dim * 2)(h)
        loc, log_std = jnp.split(out, 2, axis=-1)

        log_entropy_coefficient = self.param("log_entropy_coefficient", constant(math.log(self.init_entropy_coefficient)), (1,))
        log_kl_coefficient = self.param("log_kl_coefficient", constant(math.log(self.init_kl_coefficient)), (1,))

        return loc, log_std, log_entropy_coefficient, log_kl_coefficient


    def sample_and_log_prob(self, loc, log_std, key):
        std = jnp.exp(log_std) + self.min_std
        noise = jax.random.normal(key, shape=loc.shape)
        base_sample = loc + std * noise
        action = jnp.tanh(base_sample)
        gaussian_log_prob = -0.5 * (noise ** 2) - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(std)
        tanh_correction = 2.0 * (jnp.log(2.0) - base_sample - jax.nn.softplus(-2.0 * base_sample))
        log_prob = jnp.sum(gaussian_log_prob - tanh_correction, axis=-1)
        return action, log_prob


    def log_prob(self, loc, log_std, action):
        std = jnp.exp(log_std) + self.min_std
        eps = 1e-6
        clipped_action = jnp.clip(action, -1.0 + eps, 1.0 - eps)
        base_sample = jnp.arctanh(clipped_action)
        gaussian_log_prob = -0.5 * ((base_sample - loc) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(std)
        tanh_correction = 2.0 * (jnp.log(2.0) - base_sample - jax.nn.softplus(-2.0 * base_sample))
        return jnp.sum(gaussian_log_prob - tanh_correction, axis=-1)


    def deterministic_action(self, loc):
        return jnp.tanh(loc)


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + 0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low)
    return jax.jit(get_clipped_and_scaled_action)
