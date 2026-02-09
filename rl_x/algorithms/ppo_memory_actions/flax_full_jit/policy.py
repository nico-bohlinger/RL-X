from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))
    policy_observation_indices = jnp.concatenate([policy_observation_indices, jnp.arange(env.single_observation_space.shape[0], env.single_observation_space.shape[0] + config.algorithm.memory_action_dimension)])

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.memory_action_dimension, config.algorithm.memory_action_mean_clip, config.algorithm.std_dev, policy_observation_indices),
                get_processed_action_function(
                    config.algorithm.action_clipping_and_rescaling,
                    jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)
                ))


class Policy(nn.Module):
    as_shape: Sequence[int]
    memory_action_dimension: int
    memory_action_mean_clip: float
    std_dev: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        torso = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        torso = nn.LayerNorm()(torso)
        torso = nn.elu(torso)
        torso = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(torso)
        torso = nn.elu(torso)

        env_action_head = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(torso)
        env_action_head = nn.elu(env_action_head)
        env_action_head = nn.Dense(np.prod(self.as_shape).item(), kernel_init=orthogonal(0.01), bias_init=constant(0.0))(env_action_head)

        memory_action_head = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(torso)
        memory_action_head = nn.elu(memory_action_head)
        memory_action_head = nn.Dense(self.memory_action_dimension, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(memory_action_head)
        memory_action_head = jnp.clip(memory_action_head, -self.memory_action_mean_clip, self.memory_action_mean_clip)

        policy_mean = jnp.concatenate([env_action_head, memory_action_head], axis=-1)

        policy_logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item() + self.memory_action_dimension))

        return policy_mean, policy_logstd


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    else:
        return jax.jit(lambda x: x)
