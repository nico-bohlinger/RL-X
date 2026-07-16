
import jax
import jax.numpy as jnp
import flax
from flax import struct
from typing import NamedTuple
from flax.core import FrozenDict

@struct.dataclass
class ParamsBuffer:
    params: FrozenDict
    n: jax.Array  # index to add new entries at
    size: jax.Array # buffer size

    def serialize(self):
        serialized_params = flax.serialization.to_state_dict(self.params)
        serialized_n = np.asarray(jax.device_get(self.n))
        serialized_size = np.asarray(jax.device_get(self.size))

        return {
            "params": serialized_params,
            "n": serialized_n,
            "size": serialized_size,
        }


    @classmethod
    def create(cls, params: FrozenDict, size: int):
        return ParamsBuffer(
            params=jax.tree.map(lambda x: jnp.stack([x] * size), params),
            n=jnp.asarray(0, dtype=jnp.int32),
            size=size
        )

    @classmethod
    def add(cls, params_buffer, params: FrozenDict):
        index = params_buffer.n
        size = params_buffer.size

        def _update_buffer(buffer, new):
            def _full_size(_):
                new_last = jnp.expand_dims(new, axis=0)
                shifted = jnp.concatenate([buffer[1:], new_last], axis=0)
                return shifted
            
            def _not_full_size(_):
                return buffer.at[index].set(new)
            
            return jax.lax.cond(index < size, _not_full_size, _full_size, operand=None)

        params_updated = jax.tree.map(_update_buffer, params_buffer.params, params)    
        return params_buffer.replace(
            params=params_updated,
            n=jnp.minimum(index + 1, size),
        )

    @classmethod
    def sample(cls, params_buffer):
        level = jnp.minimum(params_buffer.n, params_buffer.size)
        return params_buffer.params, level

    @classmethod
    def sample_oldest(cls, params_buffer, k=1):
        oldest_params = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, start_index=0, slice_size=k),
            params_buffer.params
        )

        if k == 1:
            oldest_params = jax.tree.map(lambda x: x[0], oldest_params)
            
        return oldest_params


@struct.dataclass
class EtasBuffer:
    etas: jax.Array
    n: jax.Array  # index to add new entries at
    size: jax.Array # buffer size

    def serialize(self):
        serialized_etas = np.asarray(jax.device_get(self.etas))
        serialized_n = np.asarray(jax.device_get(self.n))
        serialized_size = np.asarray(jax.device_get(self.size))

        return {
            "etas": serialized_etas,
            "n": serialized_n,
            "size": serialized_size,
        }

    @classmethod
    def create(cls, etas: jax.Array, size: int):
        return EtasBuffer(
            etas=jax.tree.map(lambda x: jnp.stack([jnp.zeros_like(x)] * size), etas),
            n=jnp.asarray(0, dtype=jnp.int32),
            size=size
        )

    @classmethod
    def add(cls, etas_buffer, etas: jax.Array):
        index = etas_buffer.n
        size = etas_buffer.size

        def _update_buffer(buffer, new):
            def _full_size(_):
                new_last = jnp.expand_dims(new, axis=0)
                shifted = jnp.concatenate([buffer[1:], new_last], axis=0)
                return shifted
            
            def _not_full_size(_):
                return buffer.at[index].set(new)
            
            return jax.lax.cond(index < size, _not_full_size, _full_size, operand=None)

        etas_updated   = jax.tree.map(_update_buffer, etas_buffer.etas, etas)        
        return etas_buffer.replace(
            etas=etas_updated,
            n=jnp.minimum(index + 1, size),
        )

    @classmethod
    def sample(cls, etas_buffer):
        n = jnp.minimum(etas_buffer.n, etas_buffer.size)
        etas = etas_buffer.etas
        s = etas.shape[0]
        i = jnp.arange(s, dtype=jnp.int32)
        dest = i + (i >= n).astype(jnp.int32)

        out = jnp.zeros((s + 1,) + etas.shape[1:], dtype=etas.dtype)
        out = out.at[dest].set(etas)

        return out, n

