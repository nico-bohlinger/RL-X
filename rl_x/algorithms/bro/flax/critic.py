from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.bro.flax.layers import BroNet


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        output_nodes = config.algorithm.nr_quantiles if config.algorithm.distributional else 1
        return DoubleCritic(
            hidden_dim=config.algorithm.critic_hidden_dim,
            nr_blocks=config.algorithm.critic_nr_blocks,
            output_nodes=output_nodes,
            critic_observation_indices=critic_observation_indices,
        )


class SingleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    output_nodes: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        x = jnp.concatenate([obs, action], axis=-1)
        x = BroNet(self.hidden_dim, self.nr_blocks, output_nodes=self.output_nodes)(x)
        if self.output_nodes == 1:
            return jnp.squeeze(x, axis=-1)
        return x


class DoubleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    output_nodes: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action):
        q1 = SingleCritic(self.hidden_dim, self.nr_blocks, self.output_nodes, self.critic_observation_indices)(obs, action)
        q2 = SingleCritic(self.hidden_dim, self.nr_blocks, self.output_nodes, self.critic_observation_indices)(obs, action)
        return q1, q2
