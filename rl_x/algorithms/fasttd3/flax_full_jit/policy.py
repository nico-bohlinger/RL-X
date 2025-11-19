from typing import Sequence
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Policy(env.single_action_space.shape, policy_observation_indices)


class Policy(nn.Module):
    as_shape: Sequence[int]
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.policy_observation_indices]
        policy_mean = nn.Dense(512)(x)
        policy_mean = nn.relu(policy_mean)
        policy_mean = nn.Dense(256)(policy_mean)
        policy_mean = nn.relu(policy_mean)
        policy_mean = nn.Dense(128)(policy_mean)
        policy_mean = nn.relu(policy_mean)
        policy_mean = nn.Dense(np.prod(self.as_shape).item(), kernel_init=normal(0.01), bias_init=constant(0.0))(policy_mean)
        policy_mean = nn.tanh(policy_mean)

        return policy_mean
