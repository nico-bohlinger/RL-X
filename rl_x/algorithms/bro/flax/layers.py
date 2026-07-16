import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal


def default_kernel_init(scale=jnp.sqrt(2.0)):
    return orthogonal(scale)


class BroNet(nn.Module):
    hidden_dim: int
    nr_blocks: int
    output_nodes: int = 0

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=default_kernel_init())(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        for _ in range(self.nr_blocks):
            residual = nn.Dense(self.hidden_dim, kernel_init=default_kernel_init())(x)
            residual = nn.LayerNorm()(residual)
            residual = nn.relu(residual)
            residual = nn.Dense(self.hidden_dim, kernel_init=default_kernel_init())(residual)
            residual = nn.LayerNorm()(residual)
            x = residual + x
        if self.output_nodes > 0:
            x = nn.Dense(self.output_nodes, kernel_init=default_kernel_init())(x)
        return x
