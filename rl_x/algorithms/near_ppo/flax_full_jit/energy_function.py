import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_energyfn(config, env, ncsnv1=False):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if ncsnv1:
            return EnergyFnCond(config.algorithm.nr_hidden_units_encoder_ncsn, config.algorithm.nr_hidden_units_decoder_ncsn, config.algorithm.nr_hidden_units_encoder_ncsn[-1]//2)
        else:
            return EnergyFn(config.algorithm.nr_hidden_units_encoder_ncsn, config.algorithm.nr_hidden_units_decoder_ncsn)


class EnergyFnCond(nn.Module):
    nr_hidden_units_encoder_ncsn: Sequence[int]
    nr_hidden_units_decoder_ncsn: Sequence[int]
    half_dim: int
    steps: int = 100

    @nn.compact
    def __call__(self, x, cond):
        # Encoder
        for hidden_dim in self.nr_hidden_units_encoder_ncsn:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        # SinusoidalPosEmb
        cond = self.steps * cond
        emb = jnp.log(10000.0) / (self.half_dim - 1)
        emb = jnp.exp(-emb * jnp.arange(self.half_dim))
        emb = cond * emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        x = x + emb

        # Decoder
        for hidden_dim in self.nr_hidden_units_decoder_ncsn:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        # Final projection to scalar
        energyfn = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        energyfn = nn.elu(energyfn)

        return energyfn


class EnergyFn(nn.Module):
    nr_hidden_units_encoder_ncsn: Sequence[int]
    nr_hidden_units_decoder_ncsn: Sequence[int]

    @nn.compact
    def __call__(self, x, cond):
        x_init = x

        # Encoder
        for hidden_dim in self.nr_hidden_units_encoder_ncsn:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        # Residual 
        x = x + nn.Dense(self.nr_hidden_units_encoder_ncsn[-1], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x_init)

        # Decoder
        for hidden_dim in self.nr_hidden_units_decoder_ncsn:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.LayerNorm()(x)
            x = nn.gelu(x)

        # Final projection to scalar
        energyfn = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        energyfn = nn.gelu(energyfn)
        energyfn = energyfn / cond

        return energyfn