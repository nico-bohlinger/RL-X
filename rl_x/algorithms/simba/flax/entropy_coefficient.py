import jax.numpy as jnp
import flax.linen as nn


class EntropyCoefficient(nn.Module):
    initial_value: float = 0.01

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", init_fn=lambda key: jnp.full((), jnp.log(self.initial_value)))
        return jnp.exp(log_alpha)
