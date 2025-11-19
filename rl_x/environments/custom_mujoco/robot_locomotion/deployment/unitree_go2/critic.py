import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


def get_critic():
    return Critic()


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        critic = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = nn.LayerNorm()(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = nn.elu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(critic)
        return critic
