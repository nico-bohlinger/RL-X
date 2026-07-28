import math
import jax.numpy as jnp
import flax.linen as nn


class EntropyCoefficient(nn.Module):
    initial_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_entropy_coefficient = self.param("log_entropy_coefficient", init_fn=lambda key: jnp.full((), math.log(self.initial_value)))
        return jnp.exp(log_entropy_coefficient)


class Adjustment(nn.Module):
    init_value: float
    log_value_min: float
    log_value_max: float

    @nn.compact
    def __call__(self):
        log_value = self.param("log_value", init_fn=lambda key: jnp.full((), math.log(self.init_value)))
        log_value = self.log_value_min + (self.log_value_max - self.log_value_min) * 0.5 * (1.0 + jnp.tanh(log_value))
        return jnp.exp(log_value)


def calculate_init_log_param(value, log_value_min, log_value_max):
    ratio = (math.log(value) - log_value_min) / ((log_value_max - log_value_min) * 0.5) - 1.0
    return math.exp(math.atanh(ratio))
