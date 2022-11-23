from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_agent(config, env):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Agent(env.get_single_action_space_shape(), config.algorithm.nr_hidden_units),
                get_processed_action_function(jnp.array(env.action_space.low), jnp.array(env.action_space.high)))
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class Agent(nn.Module):
    as_shape: Sequence[int]
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x):
        torso = nn.Dense(self.nr_hidden_units * 2)(x)
        torso = nn.relu(torso)
        torso = nn.Dense(self.nr_hidden_units)(torso)
        torso = nn.relu(torso)

        policy_head = nn.Dense(self.nr_hidden_units)(torso)
        policy_head = nn.relu(policy_head)
        policy_head = nn.Dense(np.prod(self.as_shape).item() * 2)(policy_head)

        value_head = nn.Dense(self.nr_hidden_units)(torso)
        value_head = nn.relu(value_head)
        value_head = nn.Dense(1)(value_head)

        return policy_head, value_head


def get_processed_action_function(env_as_low, env_as_high):
    def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
        clipped_action = jnp.clip(action, -1, 1)
        return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
    return jax.jit(get_clipped_and_scaled_action)
