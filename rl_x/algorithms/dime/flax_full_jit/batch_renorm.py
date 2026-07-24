from typing import Any, Callable, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from jax.nn import initializers
from flax.linen.normalization import (
    _canonicalize_axes,
    _compute_stats,
    _normalize,
)
from flax.linen.module import Module, compact, merge_param


PRNGKey = Any
Array = Any
Shape = Tuple[int, ...]
Dtype = Any
Axes = Union[int, Sequence[int]]


class BatchRenorm(Module):
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    warm_up_steps: int = 100_000
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        use_running_average = merge_param(
            "use_running_average",
            self.use_running_average,
            use_running_average,
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(
            axis for axis in range(x.ndim) if axis not in feature_axes
        )
        feature_shape = [x.shape[axis] for axis in feature_axes]
        running_mean = self.variable(
            "batch_stats",
            "mean",
            lambda shape: jnp.zeros(shape, jnp.float32),
            feature_shape,
        )
        running_variance = self.variable(
            "batch_stats",
            "var",
            lambda shape: jnp.ones(shape, jnp.float32),
            feature_shape,
        )
        maximum_r = self.variable(
            "batch_stats", "r_max", lambda value: value, 3
        )
        maximum_d = self.variable(
            "batch_stats", "d_max", lambda value: value, 5
        )
        steps = self.variable(
            "batch_stats", "steps", lambda value: value, 0
        )
        if use_running_average:
            mean = running_mean.value
            variance = running_variance.value
            normalized_mean = mean
            normalized_variance = variance
        else:
            mean, variance = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=(
                    self.axis_name
                    if not self.is_initializing()
                    else None
                ),
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
            )
            normalized_mean = mean
            normalized_variance = variance
            if not self.is_initializing():
                standard_deviation = jnp.sqrt(
                    variance + self.epsilon
                )
                running_standard_deviation = jnp.sqrt(
                    running_variance.value + self.epsilon
                )
                r = jax.lax.stop_gradient(
                    standard_deviation
                    / running_standard_deviation
                )
                r = jnp.clip(
                    r,
                    1.0 / maximum_r.value,
                    maximum_r.value,
                )
                d = jax.lax.stop_gradient(
                    (mean - running_mean.value)
                    / running_standard_deviation
                )
                d = jnp.clip(
                    d, -maximum_d.value, maximum_d.value
                )
                corrected_variance = variance / r ** 2
                corrected_mean = (
                    mean
                    - d
                    * jnp.sqrt(normalized_variance)
                    / r
                )
                warmed_up = jnp.greater_equal(
                    steps.value, self.warm_up_steps
                ).astype(jnp.float32)
                normalized_variance = (
                    warmed_up * corrected_variance
                    + (1.0 - warmed_up)
                    * normalized_variance
                )
                normalized_mean = (
                    warmed_up * corrected_mean
                    + (1.0 - warmed_up)
                    * normalized_mean
                )
                running_mean.value = (
                    self.momentum * running_mean.value
                    + (1.0 - self.momentum) * mean
                )
                running_variance.value = (
                    self.momentum * running_variance.value
                    + (1.0 - self.momentum) * variance
                )
                steps.value += 1
        return _normalize(
            self,
            x,
            normalized_mean,
            normalized_variance,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
