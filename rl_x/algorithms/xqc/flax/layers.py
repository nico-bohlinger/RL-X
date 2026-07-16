import flax
import flax.linen as nn
import flax.traverse_util
import jax.numpy as jnp
from flax.linen.initializers import orthogonal


def default_kernel_init(scale=jnp.sqrt(2)):
    return orthogonal(scale)


class BNEmbedder(nn.Module):
    momentum: float = 0.99
    epsilon: float = 1e-3

    @nn.compact
    def __call__(self, x, train):
        return nn.BatchNorm(use_running_average=not train, momentum=self.momentum, epsilon=self.epsilon)(x)


class XQCBlock(nn.Module):
    hidden_dim: int
    skip_connections: bool = False
    momentum: float = 0.99
    epsilon: float = 1e-3

    @nn.compact
    def __call__(self, x, train):
        residual = x
        x = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=default_kernel_init())(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=self.momentum, epsilon=self.epsilon)(x)
        x = nn.relu(x)
        if self.skip_connections and residual.shape == x.shape:
            x = x + residual
        return x


def norm_dense_layer(params, path, norm_bias):
    kernel = params[path + "/kernel"]
    bias = params.get(path + "/bias", None)
    if norm_bias and bias is not None:
        weights = jnp.concatenate([kernel, jnp.expand_dims(bias, -2)], axis=-2)
    else:
        weights = kernel
    norm = jnp.linalg.norm(weights, axis=-2, keepdims=True)
    params[path + "/kernel"] = kernel / norm
    if norm_bias and bias is not None:
        params[path + "/bias"] = bias / norm.squeeze(-2)
    return params


def weight_norm_params(params, hidden_marker, predictor_marker, normalize_last_layer):
    params_flat = flax.traverse_util.flatten_dict(params, sep="/")
    paths = sorted({"/".join(key.split("/")[:-1]) for key in params_flat})
    for path in paths:
        if hidden_marker in path and "Dense" in path:
            params_flat = norm_dense_layer(params_flat, path, norm_bias=True)
        elif predictor_marker in path and normalize_last_layer:
            params_flat = norm_dense_layer(params_flat, path, norm_bias=False)
    return flax.traverse_util.unflatten_dict(params_flat, sep="/")
