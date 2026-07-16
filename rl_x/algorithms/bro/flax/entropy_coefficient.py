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
    init_value: float = 1.0
    log_val_min: float = -10.0
    log_val_max: float = 7.5

    @nn.compact
    def __call__(self):
        log_value = self.param("log_value", init_fn=lambda key: jnp.full((), math.log(self.init_value)))
        log_value = self.log_val_min + (self.log_val_max - self.log_val_min) * 0.5 * (1.0 + jnp.tanh(log_value))
        return jnp.exp(log_value)


def calculate_init_log_param(value, log_val_min, log_val_max):
    ratio = (math.log(value) - log_val_min) / ((log_val_max - log_val_min) * 0.5) - 1.0
    return math.exp(math.atanh(ratio))
