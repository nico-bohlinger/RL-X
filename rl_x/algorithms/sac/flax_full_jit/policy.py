from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.log_std_min, config.algorithm.log_std_max),
                get_processed_action_function(jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)))



class Policy(nn.Module):
    as_shape: Sequence[int]
    log_std_min: float
    log_std_max: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)
        x = nn.Dense(256)(x)
        x = nn.elu(x)
        x = nn.Dense(128)(x)
        x = nn.elu(x)

        mean = nn.Dense(np.prod(self.as_shape).item())(x)
        log_std = nn.Dense(np.prod(self.as_shape).item())(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
