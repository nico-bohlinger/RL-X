from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DiffusionPolicy(env.single_action_space.shape[0], config.algorithm.timestep_embed_dim, config.algorithm.policy_hidden_dims, config.algorithm.policy_output_scale, policy_observation_indices)


class DiffusionPolicy(nn.Module):
    action_dimension: int
    timestep_embed_dim: int
    hidden_dims: Sequence[int]
    output_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, noisy_action, timestep):
        observation = observation[..., self.policy_observation_indices]
        frequencies = jnp.exp(-jnp.log(10000.0) * jnp.arange(self.timestep_embed_dim // 2) / (self.timestep_embed_dim // 2 - 1))
        scaled_timestep = timestep * frequencies
        timestep_embedding = jnp.concatenate([jnp.sin(scaled_timestep), jnp.cos(scaled_timestep)], axis=-1)
        timestep_embedding = nn.Dense(2 * self.timestep_embed_dim)(timestep_embedding)
        timestep_embedding = timestep_embedding * jnp.tanh(jax.nn.softplus(timestep_embedding))
        timestep_embedding = nn.Dense(self.timestep_embed_dim)(timestep_embedding)
        x = jnp.concatenate([noisy_action, timestep_embedding, observation], axis=-1)
        x = nn.Dense(self.hidden_dims[0], kernel_init=nn.initializers.lecun_uniform())(x)
        for hidden_index in range(1, len(self.hidden_dims), 2):
            residual = x
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dims[hidden_index], kernel_init=nn.initializers.lecun_uniform())(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dims[hidden_index + 1], kernel_init=nn.initializers.lecun_uniform())(x)
            x = x + residual
        x = nn.Dense(self.action_dimension, kernel_init=nn.initializers.lecun_uniform())(x)
        return x * self.output_scale
