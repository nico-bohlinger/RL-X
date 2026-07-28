import jax
import jax.numpy as jnp
import flax.linen as nn


class BatchRenorm(nn.Module):
    momentum: float
    warm_up_steps: int
    epsilon: float = 0.001

    @nn.compact
    def __call__(self, x, train):
        running_mean = self.variable("batch_stats", "mean", lambda: jnp.zeros(x.shape[-1], dtype=jnp.float32))
        running_variance = self.variable("batch_stats", "variance", lambda: jnp.ones(x.shape[-1], dtype=jnp.float32))
        steps = self.variable("batch_stats", "steps", lambda: jnp.zeros((), dtype=jnp.int32))
        scale = self.param("scale", nn.initializers.ones_init(), (x.shape[-1],))
        bias = self.param("bias", nn.initializers.zeros_init(), (x.shape[-1],))

        if train:
            reduction_axes = tuple(range(x.ndim - 1))
            mean = jnp.mean(x, axis=reduction_axes)
            variance = jnp.var(x, axis=reduction_axes)
            standard_deviation = jnp.sqrt(variance + self.epsilon)
            running_standard_deviation = jnp.sqrt(running_variance.value + self.epsilon)
            r = jax.lax.stop_gradient(jnp.clip(standard_deviation / running_standard_deviation, 1.0 / 3.0, 3.0))
            d = jax.lax.stop_gradient(jnp.clip((mean - running_mean.value) / running_standard_deviation, -5.0, 5.0))
            normalized = (x - mean) / standard_deviation
            normalized = jnp.where(steps.value >= self.warm_up_steps, normalized * r + d, normalized)
            if not self.is_initializing():
                running_mean.value = self.momentum * running_mean.value + (1.0 - self.momentum) * mean
                running_variance.value = self.momentum * running_variance.value + (1.0 - self.momentum) * variance
                steps.value += 1
        else:
            normalized = (x - running_mean.value) / jnp.sqrt(running_variance.value + self.epsilon)

        return normalized * scale + bias
