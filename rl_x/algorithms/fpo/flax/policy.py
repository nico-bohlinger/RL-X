from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return FlowPolicy(env.single_action_space.shape[0], config.algorithm.timestep_embed_dim, config.algorithm.policy_hidden_dims, config.algorithm.policy_output_scale, policy_observation_indices)


class FlowPolicy(nn.Module):
    action_dimension: int
    timestep_embed_dim: int
    hidden_dims: Sequence[int]
    output_scale: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, noisy_action, timestep):
        observation = observation[..., self.policy_observation_indices]
        frequencies = 2 ** jnp.arange(self.timestep_embed_dim // 2)
        scaled_timestep = timestep * frequencies
        timestep_embedding = jnp.concatenate([jnp.cos(scaled_timestep), jnp.sin(scaled_timestep)], axis=-1)
        x = jnp.concatenate([observation, timestep_embedding, noisy_action], axis=-1)
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension, kernel_init=nn.initializers.lecun_uniform())(x)
            x = nn.elu(x)
        x = nn.Dense(self.action_dimension, kernel_init=nn.initializers.lecun_uniform())(x)
        return x * self.output_scale
