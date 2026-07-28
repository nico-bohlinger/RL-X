import jax.numpy as jnp
import flax.linen as nn


def get_entropy_coefficient(config):
    return EntropyCoefficient(config.algorithm.entropy_coefficient_init)


class EntropyCoefficient(nn.Module):
    initial_value: float

    @nn.compact
    def __call__(self):
        log_coefficient = self.param("log_coefficient", lambda key: jnp.asarray(jnp.log(self.initial_value)))
        return jnp.exp(log_coefficient)
