import math
import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.traverse_util


EPS = 1e-8


def l2normalize(x, axis):
    norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / jnp.maximum(norm, EPS)


class Scaler(nn.Module):
    dim: int
    init: float = 1.0
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        scaler = self.param("scaler", nn.initializers.constant(self.scale), (self.dim,))
        return scaler * (self.init / self.scale) * x


class HyperDense(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            use_bias=False,
            name="hyper_dense",
        )(x)


class HyperEmbedder(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    c_shift: float

    @nn.compact
    def __call__(self, x):
        shift_axis = jnp.ones(x.shape[:-1] + (1,)) * self.c_shift
        x = jnp.concatenate([x, shift_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = HyperDense(self.hidden_dim)(x)
        x = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperMLP(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float

    @nn.compact
    def __call__(self, x):
        x = HyperDense(self.hidden_dim)(x)
        x = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)(x)
        x = nn.relu(x) + EPS
        x = HyperDense(self.out_dim)(x)
        x = l2normalize(x, axis=-1)
        return x


class HyperLERPBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    expansion: int = 4

    @nn.compact
    def __call__(self, x):
        residual = x
        x = HyperMLP(
            hidden_dim=self.hidden_dim * self.expansion,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init / math.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / math.sqrt(self.expansion),
        )(x)
        alpha = Scaler(self.hidden_dim, self.alpha_init, self.alpha_scale)(x - residual)
        x = residual + alpha
        x = l2normalize(x, axis=-1)
        return x


class HyperNormalTanhPolicy(nn.Module):
    hidden_dim: int
    action_dim: int
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        mean = HyperDense(self.hidden_dim)(x)
        mean = Scaler(self.hidden_dim, 1.0, 1.0, name="mean_scaler")(mean)
        mean = HyperDense(self.action_dim)(mean) + self.param("mean_bias", nn.initializers.zeros, (self.action_dim,))
        log_std = HyperDense(self.hidden_dim)(x)
        log_std = Scaler(self.hidden_dim, 1.0, 1.0, name="log_std_scaler")(log_std)
        log_std = HyperDense(self.action_dim)(log_std) + self.param("log_std_bias", nn.initializers.zeros, (self.action_dim,))
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1.0 + jnp.tanh(log_std))
        return mean, log_std


class HyperCategoricalValue(nn.Module):
    hidden_dim: int
    nr_bins: int
    v_min: float
    v_max: float

    @nn.compact
    def __call__(self, x):
        v = HyperDense(self.hidden_dim)(x)
        v = Scaler(self.hidden_dim, 1.0, 1.0, name="value_scaler")(v)
        v = HyperDense(self.nr_bins)(v) + self.param("value_bias", nn.initializers.zeros, (self.nr_bins,))
        log_probs = jax.nn.log_softmax(v, axis=-1)
        bin_values = jnp.linspace(self.v_min, self.v_max, self.nr_bins, dtype=jnp.float32)
        value = jnp.sum(jnp.exp(log_probs) * bin_values, axis=-1)
        return value, log_probs


def l2normalize_params(params):
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    new_flat = {}
    for path, value in flat.items():
        if "hyper_dense" in path and path.endswith("/kernel"):
            if value.ndim == 2:
                value = l2normalize(value, axis=0)
            elif value.ndim == 3:
                value = l2normalize(value, axis=1)
            new_flat[path] = value
        else:
            new_flat[path] = value
    return flax.traverse_util.unflatten_dict(new_flat, sep="/")
