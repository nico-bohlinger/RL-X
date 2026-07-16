from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.bro.flax.layers import BroNet, default_kernel_init


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        action_dim = env.single_action_space.shape[0]
        pessimistic_policy = NormalTanhPolicy(
            action_dim=action_dim,
            hidden_dim=config.algorithm.policy_hidden_dim,
            nr_blocks=config.algorithm.policy_nr_blocks,
            policy_observation_indices=policy_observation_indices,
        )
        optimistic_policy = DualTanhPolicy(
            action_dim=action_dim,
            hidden_dim=config.algorithm.policy_hidden_dim,
            nr_blocks=config.algorithm.policy_nr_blocks,
            policy_observation_indices=policy_observation_indices,
        )
        return (
            pessimistic_policy,
            optimistic_policy,
            get_processed_action_function(jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)),
        )


class NormalTanhPolicy(nn.Module):
    action_dim: int
    hidden_dim: int
    nr_blocks: int
    policy_observation_indices: Sequence[int]
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, obs):
        obs = obs[..., self.policy_observation_indices]
        trunk = BroNet(self.hidden_dim, self.nr_blocks)(obs)
        mean = nn.Dense(self.action_dim, kernel_init=default_kernel_init())(trunk)
        log_std = nn.Dense(self.action_dim, kernel_init=default_kernel_init(1.0))(trunk)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1.0 + jnp.tanh(log_std))
        std = jnp.exp(log_std)
        return mean, std


    def sample_and_log_prob(self, mean, std, key):
        log_std = jnp.log(std)
        noise = jax.random.normal(key, shape=mean.shape)
        pretanh = mean + std * noise
        action = jnp.tanh(pretanh)
        gaussian_log_prob = -0.5 * (noise ** 2) - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
        tanh_correction = 2.0 * (jnp.log(2.0) - pretanh - jax.nn.softplus(-2.0 * pretanh))
        log_prob = jnp.sum(gaussian_log_prob - tanh_correction, axis=-1)
        return action, log_prob


    def sample_action(self, mean, std, key):
        noise = jax.random.normal(key, shape=mean.shape)
        return jnp.tanh(mean + std * noise)


    def deterministic_action(self, mean):
        return jnp.tanh(mean)


class DualTanhPolicy(nn.Module):
    action_dim: int
    hidden_dim: int
    nr_blocks: int
    policy_observation_indices: Sequence[int]
    scale_means: float = 0.01

    @nn.compact
    def __call__(self, obs, base_mean, base_std, std_multiplier):
        obs = obs[..., self.policy_observation_indices]
        x = jnp.concatenate([obs, base_mean], axis=-1)
        trunk = BroNet(self.hidden_dim, self.nr_blocks)(x)
        shift = nn.Dense(self.action_dim, kernel_init=default_kernel_init(self.scale_means), use_bias=False)(trunk)
        opt_mean = base_mean + shift
        opt_std = base_std * std_multiplier
        return opt_mean, opt_std


    def sample_and_log_prob(self, mean, std, key):
        log_std = jnp.log(std)
        noise = jax.random.normal(key, shape=mean.shape)
        pretanh = mean + std * noise
        action = jnp.tanh(pretanh)
        gaussian_log_prob = -0.5 * (noise ** 2) - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
        tanh_correction = 2.0 * (jnp.log(2.0) - pretanh - jax.nn.softplus(-2.0 * pretanh))
        log_prob = jnp.sum(gaussian_log_prob - tanh_correction, axis=-1)
        return action, log_prob


    def sample_action(self, mean, std, key):
        noise = jax.random.normal(key, shape=mean.shape)
        return jnp.tanh(mean + std * noise)


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
