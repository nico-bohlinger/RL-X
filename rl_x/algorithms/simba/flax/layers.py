import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal


class PreLNResidualBlock(nn.Module):
    hidden_dim: int
    expansion: int = 4

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim * self.expansion, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        return residual + x


class SimbaEncoder(nn.Module):
    hidden_dim: int
    nr_blocks: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        for _ in range(self.nr_blocks):
            x = PreLNResidualBlock(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        return x


class NormalTanhPolicyHead(nn.Module):
    action_dim: int
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0))(x)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1.0 + jnp.tanh(log_std))
        return mean, log_std
