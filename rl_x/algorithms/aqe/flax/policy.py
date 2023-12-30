from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from rl_x.algorithms.aqe.flax.tanh_transformed_distribution import TanhTransformedDistribution

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.log_std_min, config.algorithm.log_std_max, config.algorithm.nr_hidden_units),
                get_processed_action_function(jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)))



class Policy(nn.Module):
    as_shape: Sequence[int]
    log_std_min: float
    log_std_max: float
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)

        mean = nn.Dense(np.prod(self.as_shape).item())(x)
        log_std = nn.Dense(np.prod(self.as_shape).item())(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        dist = TanhTransformedDistribution(tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)))

        return dist


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
