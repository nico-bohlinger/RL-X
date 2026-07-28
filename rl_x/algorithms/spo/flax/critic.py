from typing import Sequence
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class Critic(nn.Module):
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x[..., self.critic_observation_indices]
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.tanh(critic)
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
