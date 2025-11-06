from typing import Sequence
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(config.algorithm.nr_hidden_units, critic_observation_indices)


class Critic(nn.Module):
    nr_hidden_units: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = x[..., self.critic_observation_indices]
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
