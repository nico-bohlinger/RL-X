from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.algorithms.dime.flax_full_jit.batch_renorm import BatchRenorm
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorDistributionalCritic(config.algorithm.nr_critics, config.algorithm.critic_hidden_dims, config.algorithm.nr_atoms, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps, critic_observation_indices)


class DistributionalCritic(nn.Module):
    hidden_dims: Sequence[int]
    nr_atoms: int
    batch_renorm_momentum: float
    batch_renorm_warmup_steps: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action, train):
        x = jnp.concatenate([observation[..., self.critic_observation_indices], action], axis=-1)
        x = BatchRenorm(self.batch_renorm_momentum, self.batch_renorm_warmup_steps)(x, train)
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension)(x)
            x = nn.relu(x)
            x = BatchRenorm(self.batch_renorm_momentum, self.batch_renorm_warmup_steps)(x, train)
        return jax.nn.softmax(nn.Dense(self.nr_atoms)(x), axis=-1)


class VectorDistributionalCritic(nn.Module):
    nr_critics: int
    hidden_dims: Sequence[int]
    nr_atoms: int
    batch_renorm_momentum: float
    batch_renorm_warmup_steps: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action, train):
        vectorized_critic = nn.vmap(DistributionalCritic, variable_axes={"params": 0, "batch_stats": 0}, split_rngs={"params": True, "batch_stats": True}, in_axes=None, out_axes=0, axis_size=self.nr_critics)
        return vectorized_critic(hidden_dims=self.hidden_dims, nr_atoms=self.nr_atoms, batch_renorm_momentum=self.batch_renorm_momentum, batch_renorm_warmup_steps=self.batch_renorm_warmup_steps, critic_observation_indices=self.critic_observation_indices)(observation, action, train)
