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
        return ScorePolicy(env.single_action_space.shape[0], config.algorithm.timestep_embed_dim, config.algorithm.score_hidden_dims, config.algorithm.score_output_scale, config.algorithm.initial_timestep, config.algorithm.initial_friction, policy_observation_indices)


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
        timestep_embedding = jnp.concatenate([jnp.sin(timestep_coefficients * timestep + timestep_phase), jnp.cos(timestep_coefficients * timestep + timestep_phase)], axis=-1)
        timestep_embedding = nn.Dense(self.timestep_embed_dim)(timestep_embedding)
        timestep_embedding = nn.gelu(timestep_embedding)
        timestep_embedding = nn.Dense(self.timestep_embed_dim)(timestep_embedding)
        x = jnp.concatenate([action, observation, timestep_embedding], axis=-1)
        for hidden_dimension in self.hidden_dims:
            x = nn.Dense(hidden_dimension)(x)
            x = nn.gelu(x)
        return jnp.clip(nn.Dense(self.action_dimension, kernel_init=nn.initializers.constant(self.output_scale), bias_init=nn.initializers.zeros_init())(x), -1e4, 1e4)
