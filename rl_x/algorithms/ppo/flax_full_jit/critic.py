from typing import Sequence
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        nr_q_critics = getattr(config.algorithm, "nr_q_critics", 1)
        if nr_q_critics > 1:
            return VectorCritic(nr_q_critics, critic_observation_indices)
        return Critic(critic_observation_indices, getattr(config.algorithm, "nr_value_samples", 0) > 0)


class Critic(nn.Module):
    critic_observation_indices: Sequence[int]
    action_conditioned: bool

    @nn.compact
    def __call__(self, x, action=None):
        x = x[..., self.critic_observation_indices]
        if self.action_conditioned:
            x = jnp.concatenate([x, action], axis=-1)
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.LayerNorm()(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic)
        return critic


class VectorCritic(nn.Module):
    nr_critics: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x, action):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )
        return vmap_critic(critic_observation_indices=self.critic_observation_indices, action_conditioned=True)(x, action)
