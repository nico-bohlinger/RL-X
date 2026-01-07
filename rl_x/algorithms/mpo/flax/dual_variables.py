import jax.numpy as jnp
import flax.linen as nn


class DualVariables(nn.Module):
    nr_actions: int
    init_log_eta: float
    init_log_alpha_mean: float
    init_log_alpha_stddev: float
    init_log_penalty_temperature: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_eta = self.param("log_eta", init_fn=lambda key: jnp.full([1], self.init_log_eta, dtype=jnp.float32))
        log_alpha_mean = self.param("log_alpha_mean", init_fn=lambda key: jnp.full([self.nr_actions], self.init_log_alpha_mean, dtype=jnp.float32))
        log_alpha_stddev = self.param("log_alpha_stddev", init_fn=lambda key: jnp.full([self.nr_actions], self.init_log_alpha_stddev, dtype=jnp.float32))
        log_penalty_temperature = self.param("log_penalty_temperature", init_fn=lambda key: jnp.full([1], self.init_log_penalty_temperature, dtype=jnp.float32))
        return log_eta, log_alpha_mean, log_alpha_stddev, log_penalty_temperature
