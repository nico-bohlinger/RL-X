import jax.numpy as jnp
import flax.linen as nn


class EntropyCoefficient(nn.Module):
    init_ent_coef: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_alpha = self.param("log_alpha", init_fn=lambda key: jnp.full((), jnp.log(self.init_ent_coef)))
        return jnp.exp(log_alpha)


class ConstantEntropyCoefficient(nn.Module):
    init_ent_coef: float

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.init_ent_coef))
        return self.init_ent_coef
