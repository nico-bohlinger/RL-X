from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.simba.flax.layers import SimbaEncoder


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        critic_class = DoubleCritic if config.algorithm.use_cdq else SingleCritic
        return critic_class(
            hidden_dim=config.algorithm.critic_hidden_dim,
            nr_blocks=config.algorithm.critic_nr_blocks,
            critic_observation_indices=critic_observation_indices,
        )


class SingleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        x = jnp.concatenate([obs, action], axis=-1)
        x = SimbaEncoder(self.hidden_dim, self.nr_blocks)(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return jnp.squeeze(value, axis=-1)


class DoubleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SingleCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        return vmap_critic(
            hidden_dim=self.hidden_dim,
            nr_blocks=self.nr_blocks,
            critic_observation_indices=self.critic_observation_indices,
        )(obs, action)
