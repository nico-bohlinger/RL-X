from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.flashsac.flax.layers import (
    EnsembleCategoricalValue,
    FlashSACBlock,
    FlashSACEmbedder,
    UnitRMSNorm,
)


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DoubleCritic(
            hidden_dim=config.algorithm.critic_hidden_dim,
            nr_blocks=config.algorithm.critic_nr_blocks,
            nr_atoms=config.algorithm.nr_atoms,
            v_min=config.algorithm.v_min,
            v_max=config.algorithm.v_max,
            critic_observation_indices=critic_observation_indices,
        )


class SingleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    nr_atoms: int
    v_min: float
    v_max: float
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action, train):
        obs = obs[..., self.critic_observation_indices]
        x = jnp.concatenate([obs, action], axis=-1)
        x = FlashSACEmbedder(self.hidden_dim)(x, train)
        for _ in range(self.nr_blocks):
            x = FlashSACBlock(self.hidden_dim)(x, train)
        x = UnitRMSNorm()(x)
        value, log_probs = EnsembleCategoricalValue(self.nr_atoms, self.v_min, self.v_max)(x)
        return value, log_probs


class DoubleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    nr_atoms: int
    v_min: float
    v_max: float
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action, train):
        vmap_critic = nn.vmap(
            SingleCritic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        values, log_probs = vmap_critic(
            hidden_dim=self.hidden_dim,
            nr_blocks=self.nr_blocks,
            nr_atoms=self.nr_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            critic_observation_indices=self.critic_observation_indices,
        )(obs, action, train)
        return values, log_probs
