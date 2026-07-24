from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn


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
        x = jnp.concatenate([observation, noisy_action, timestep_embedding], axis=-1)
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension, kernel_init=nn.initializers.lecun_uniform())(x)
            x = nn.silu(x)
        x = nn.Dense(self.action_dimension, kernel_init=nn.initializers.lecun_uniform())(x)
        return x * self.output_scale


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation):
        x = observation[..., self.critic_observation_indices]
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension, kernel_init=nn.initializers.lecun_uniform())(x)
            x = nn.silu(x)
        return nn.Dense(1, kernel_init=nn.initializers.lecun_uniform())(x)
