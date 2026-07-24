from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


class ScorePolicy(nn.Module):
    action_dimension: int
    timestep_embed_dim: int
    hidden_dims: Sequence[int]
    output_scale: float
    initial_timestep: float
    initial_friction: float
    policy_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action, timestep):
        self.param(
            "log_timestep",
            lambda key: jnp.asarray(jnp.log(jnp.expm1(self.initial_timestep))),
        )
        self.param(
            "log_friction",
            lambda key: jnp.asarray(jnp.log(jnp.expm1(self.initial_friction))),
        )
        observation = observation[..., self.policy_observation_indices]
        frequencies = 2 ** jnp.arange(self.timestep_embed_dim // 2)
        timestep_embedding = jnp.concatenate(
            [
                jnp.cos(timestep * frequencies),
                jnp.sin(timestep * frequencies),
            ],
            axis=-1,
        )
        x = jnp.concatenate(
            [observation, action, timestep_embedding], axis=-1
        )
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension)(x)
            x = nn.swish(x)
        return self.output_scale * nn.Dense(self.action_dimension)(x)


class DistributionalCritic(nn.Module):
    hidden_dims: Sequence[int]
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action):
        x = jnp.concatenate(
            [
                observation[..., self.critic_observation_indices],
                action,
            ],
            axis=-1,
        )
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension)(x)
            x = nn.relu(x)
        return jax.nn.softmax(nn.Dense(self.nr_atoms)(x), axis=-1)


class VectorDistributionalCritic(nn.Module):
    nr_critics: int
    hidden_dims: Sequence[int]
    nr_atoms: int
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, observation, action):
        vectorized_critic = nn.vmap(
            DistributionalCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )
        return vectorized_critic(
            hidden_dims=self.hidden_dims,
            nr_atoms=self.nr_atoms,
            critic_observation_indices=self.critic_observation_indices,
        )(observation, action)


class EntropyCoefficient(nn.Module):
    initial_value: float

    @nn.compact
    def __call__(self):
        log_coefficient = self.param(
            "log_coefficient",
            lambda key: jnp.asarray(jnp.log(self.initial_value)),
        )
        return jnp.exp(log_coefficient)
