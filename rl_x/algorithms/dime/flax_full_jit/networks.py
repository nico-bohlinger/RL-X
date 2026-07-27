from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

from rl_x.algorithms.dime.flax_full_jit.batch_renorm import BatchRenorm


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
        self.param("log_timestep", lambda key: jnp.full((1,), jnp.log(jnp.expm1(self.initial_timestep))))
        self.param("log_friction", lambda key: jnp.full((self.action_dimension,), jnp.log(jnp.expm1(self.initial_friction))))
        observation = observation[..., self.policy_observation_indices]
        timestep_phase = self.param("timestep_phase", nn.initializers.zeros_init(), (1, self.timestep_embed_dim))
        timestep_coefficients = jnp.linspace(0.1, 100.0, self.timestep_embed_dim)[None]
        timestep_embedding = jnp.concatenate(
            [
                jnp.sin(timestep_coefficients * timestep + timestep_phase),
                jnp.cos(timestep_coefficients * timestep + timestep_phase),
            ],
            axis=-1,
        )
        timestep_embedding = nn.Dense(self.timestep_embed_dim)(timestep_embedding)
        timestep_embedding = nn.gelu(timestep_embedding)
        timestep_embedding = nn.Dense(self.timestep_embed_dim)(timestep_embedding)
        x = jnp.concatenate([action, observation, timestep_embedding], axis=-1)
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension)(x)
            x = nn.gelu(x)
        return jnp.clip(
            nn.Dense(self.action_dimension, kernel_init=nn.initializers.constant(self.output_scale), bias_init=nn.initializers.zeros_init())(x),
            -1e4,
            1e4,
        )


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
        vectorized_critic = nn.vmap(
            DistributionalCritic,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={"params": True, "batch_stats": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )
        return vectorized_critic(
            hidden_dims=self.hidden_dims,
            nr_atoms=self.nr_atoms,
            batch_renorm_momentum=self.batch_renorm_momentum,
            batch_renorm_warmup_steps=self.batch_renorm_warmup_steps,
            critic_observation_indices=self.critic_observation_indices,
        )(observation, action, train)


class EntropyCoefficient(nn.Module):
    initial_value: float

    @nn.compact
    def __call__(self):
        log_coefficient = self.param("log_coefficient", lambda key: jnp.asarray(jnp.log(self.initial_value)))
        return jnp.exp(log_coefficient)
