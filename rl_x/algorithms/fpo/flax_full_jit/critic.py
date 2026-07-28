from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return ValueCritic(config.algorithm.critic_hidden_dims, critic_observation_indices)


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation):
        x = observation[..., self.critic_observation_indices]
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension, kernel_init=nn.initializers.lecun_uniform())(x)
            x = nn.elu(x)
        return nn.Dense(1, kernel_init=nn.initializers.lecun_uniform())(x)
