import math
from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.simbav2.flax.layers import HyperEmbedder, HyperLERPBlock, HyperCategoricalValue


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        hidden_dim = config.algorithm.critic_hidden_dim
        nr_blocks = config.algorithm.critic_nr_blocks
        scaler_init = math.sqrt(2.0 / hidden_dim)
        scaler_scale = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (nr_blocks + 1)
        alpha_scale = 1.0 / math.sqrt(hidden_dim)
        critic_class = DoubleCritic if config.algorithm.use_cdq else SingleCritic
        return critic_class(
            hidden_dim=hidden_dim,
            nr_blocks=nr_blocks,
            c_shift=config.algorithm.c_shift,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            nr_bins=config.algorithm.nr_bins,
            v_min=config.algorithm.v_min,
            v_max=config.algorithm.v_max,
            critic_observation_indices=critic_observation_indices,
        )


class SingleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    c_shift: float
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    nr_bins: int
    v_min: float
    v_max: float
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        x = jnp.concatenate([obs, action], axis=-1)
        x = HyperEmbedder(self.hidden_dim, self.scaler_init, self.scaler_scale, self.c_shift)(x)
        for _ in range(self.nr_blocks):
            x = HyperLERPBlock(self.hidden_dim, self.scaler_init, self.scaler_scale, self.alpha_init, self.alpha_scale)(x)
        value, log_probs = HyperCategoricalValue(self.hidden_dim, self.nr_bins, self.v_min, self.v_max)(x)
        return value, log_probs


class DoubleCritic(nn.Module):
    hidden_dim: int
    nr_blocks: int
    c_shift: float
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    nr_bins: int
    v_min: float
    v_max: float
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SingleCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=2,
        )
        values, log_probs = vmap_critic(
            hidden_dim=self.hidden_dim,
            nr_blocks=self.nr_blocks,
            c_shift=self.c_shift,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            nr_bins=self.nr_bins,
            v_min=self.v_min,
            v_max=self.v_max,
            critic_observation_indices=self.critic_observation_indices,
        )(obs, action)
        return values, log_probs
