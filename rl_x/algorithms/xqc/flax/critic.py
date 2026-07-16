from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.xqc.flax.layers import BNEmbedder, XQCBlock, default_kernel_init


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorCritic(
            hidden_dim=config.algorithm.critic_hidden_dim,
            nr_blocks=config.algorithm.critic_nr_blocks,
            nr_atoms=config.algorithm.nr_atoms,
            v_min=config.algorithm.v_min,
            v_max=config.algorithm.v_max,
            nr_critics=config.algorithm.nr_critics,
            skip_connections=config.algorithm.skip_connections,
            critic_observation_indices=critic_observation_indices,
        )


class Critic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    nr_atoms: int
    v_min: float
    v_max: float
    skip_connections: bool
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action, train):
        obs = obs[..., self.critic_observation_indices]
        x = jnp.concatenate([obs, action], axis=-1)
        x = BNEmbedder(name="embedder")(x, train)
        for i in range(self.nr_blocks):
            x = XQCBlock(self.hidden_dim, self.skip_connections, name=f"block_{i}")(x, train)
        logits = nn.Dense(self.nr_atoms, use_bias=True, kernel_init=default_kernel_init(), name="value")(x)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        bin_values = jnp.linspace(self.v_min, self.v_max, self.nr_atoms, dtype=jnp.float32)
        value = jnp.sum(jnp.exp(log_probs) * bin_values, axis=-1)
        return value, log_probs


class VectorCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    nr_atoms: int
    v_min: float
    v_max: float
    nr_critics: int
    skip_connections: bool
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action, train):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )
        values, log_probs = vmap_critic(
            hidden_dim=self.hidden_dim,
            nr_blocks=self.nr_blocks,
            nr_atoms=self.nr_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            skip_connections=self.skip_connections,
            critic_observation_indices=self.critic_observation_indices,
        )(obs, action, train)
        return values, log_probs
