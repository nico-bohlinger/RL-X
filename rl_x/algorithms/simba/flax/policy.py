from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.simba.flax.layers import SimbaEncoder, NormalTanhPolicyHead


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (
            Policy(
                action_dim=env.single_action_space.shape[0],
                hidden_dim=config.algorithm.policy_hidden_dim,
                nr_blocks=config.algorithm.policy_nr_blocks,
                log_std_min=config.algorithm.log_std_min,
                log_std_max=config.algorithm.log_std_max,
                policy_observation_indices=policy_observation_indices,
            ),
            get_processed_action_function(jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)),
        )


class Policy(nn.Module):
    action_dim: int
    hidden_dim: int
    nr_blocks: int
    log_std_min: float
    log_std_max: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        x = SimbaEncoder(self.hidden_dim, self.nr_blocks)(x)
        mean, log_std = NormalTanhPolicyHead(self.action_dim, self.log_std_min, self.log_std_max)(x)
        return mean, log_std


    def sample_and_log_prob(self, mean, log_std, key):
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, shape=mean.shape)
        pretanh = mean + std * noise
        action = jnp.tanh(pretanh)
        gaussian_log_prob = -0.5 * (noise ** 2) - 0.5 * jnp.log(2.0 * jnp.pi) - log_std
        tanh_correction = 2.0 * (jnp.log(2.0) - pretanh - jax.nn.softplus(-2.0 * pretanh))
        log_prob = jnp.sum(gaussian_log_prob - tanh_correction, axis=-1)
        return action, log_prob


    def sample_action(self, mean, log_std, key):
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, shape=mean.shape)
        return jnp.tanh(mean + std * noise)


    def deterministic_action(self, mean):
        return jnp.tanh(mean)


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
