import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant


class UnitLinear(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", orthogonal(1.0), (x.shape[-1], self.features))
        return jnp.dot(x, kernel)


class UnitBatchNorm(nn.Module):
    momentum: float = 0.99
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, train):
        return nn.BatchNorm(use_running_average=not train, momentum=self.momentum, epsilon=self.epsilon)(x)


class UnitRMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        return nn.RMSNorm(epsilon=self.epsilon)(x)


class FlashSACEmbedder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, train):
        x = UnitBatchNorm()(x, train)
        x = UnitLinear(self.hidden_dim)(x)
        return x


class FlashSACBlock(nn.Module):
    hidden_dim: int
    expansion: int = 4

    @nn.compact
    def __call__(self, x, train):
        residual = x
        x = UnitLinear(self.hidden_dim * self.expansion)(x)
        x = UnitBatchNorm()(x, train)
        x = nn.relu(x)
        x = UnitLinear(self.hidden_dim)(x)
        x = UnitBatchNorm()(x, train)
        x = nn.relu(x)
        return x + residual


class NormalTanhPolicy(nn.Module):
    action_dim: int
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x):
        mean_kernel = self.param("mean_kernel", orthogonal(1.0), (x.shape[-1], self.action_dim))
        mean_bias = self.param("mean_bias", constant(0.0), (self.action_dim,))
        std_kernel = self.param("std_kernel", orthogonal(1.0), (x.shape[-1], self.action_dim))
        std_bias = self.param("std_bias", constant(0.0), (self.action_dim,))

        mean = jnp.dot(x, mean_kernel) + mean_bias
        raw_log_std = jnp.dot(x, std_kernel) + std_bias
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1.0 + jnp.tanh(raw_log_std))
        return mean, jnp.exp(log_std)


class EnsembleCategoricalValue(nn.Module):
    nr_atoms: int
    v_min: float
    v_max: float

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", orthogonal(1.0), (x.shape[-1], self.nr_atoms))
        bias = self.param("bias", constant(0.0), (self.nr_atoms,))
        logits = jnp.dot(x, kernel) + bias
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        bin_values = jnp.linspace(self.v_min, self.v_max, self.nr_atoms, dtype=jnp.float32)
        value = jnp.sum(jnp.exp(log_probs) * bin_values, axis=-1)
        return value, log_probs


def project_params(params):
    def project_leaf(path, leaf):
        name = path[-1].key if path else ""
        parent = path[-2].key if len(path) >= 2 else ""
        grandparent = path[-3].key if len(path) >= 3 else ""

        if "BatchNorm" in parent or "BatchNorm" in grandparent:
            return leaf
        if "RMSNorm" in parent or "RMSNorm" in grandparent:
            if name == "scale":
                dimension = leaf.shape[-1]
                squared_sum = jnp.sum(leaf * leaf, axis=-1, keepdims=True)
                factor = math.sqrt(dimension) * jax.lax.rsqrt(squared_sum + 1e-8)
                return leaf * factor
            return leaf
        if name in ("kernel", "mean_kernel", "std_kernel"):
            norm = jnp.linalg.norm(leaf, axis=-2, keepdims=True)
            return leaf / jnp.where(norm < 1e-8, 1.0, norm)
        return leaf

    params = jax.tree_util.tree_map_with_path(project_leaf, params, is_leaf=lambda leaf: isinstance(leaf, jnp.ndarray))

    def normalize_batchnorm_affine(parameter_dict, parent_name=""):
        if not isinstance(parameter_dict, dict):
            return parameter_dict
        if "scale" in parameter_dict and "bias" in parameter_dict and isinstance(parameter_dict["scale"], jnp.ndarray) and isinstance(parameter_dict["bias"], jnp.ndarray) and "BatchNorm" in parent_name:
            scale, bias = parameter_dict["scale"], parameter_dict["bias"]
            dim = scale.shape[-1]
            squared_sum = jnp.sum(scale * scale + bias * bias, axis=-1, keepdims=True)
            factor = math.sqrt(dim) * jax.lax.rsqrt(squared_sum + 1e-8)
            return {**parameter_dict, "scale": scale * factor, "bias": bias * factor}
        return {key: normalize_batchnorm_affine(value, parent_name=key) for key, value in parameter_dict.items()}

    return normalize_batchnorm_affine(params)
