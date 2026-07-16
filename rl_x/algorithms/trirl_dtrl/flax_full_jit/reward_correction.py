import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from functools import partial
from jax.tree_util import tree_map, tree_leaves

####################################################################
####################################################################
"""
Chunked Reward Correction
"""
####################################################################
####################################################################

def make_chunked_ensemble_rew_correct(
    log_density_ratio_fn, nr_steps: int, nr_envs: int,
    epsilon: float, beta: float, entropy_coef: float, maximum_eta: bool
):
    """
    Only load a fixed-size chunk of the buffer into memory. Avoid OOM issues and allows deepr discriminator chains

    Returns:
    chunked_correct(buffer, inputs, etas, nr_corrections, chunk_size=None) -> corr_reward (nr_steps, nr_envs)

    Args:
        buffer: python list/tuple of param pytrees (length B, dynamic)
        inputs: (states, actions) with states/actions shaped (N, ...) where N = nr_steps * nr_envs
        etas: jnp array shape (B, nr_steps, nr_envs)
        nr_corrections: number of models to correct through (<= B)
    """

    vmapped_chunk = jax.vmap(
        log_density_ratio_fn,
        in_axes=(None, 0),  # (inputs, params_batch)
        out_axes=0,
    )

    @partial(jax.jit, static_argnames=('maximum_eta', 'chunk_size'))
    def process_chunk_on_device(
        corr, params_chunk, eta_chunk, inputs,
        maximum_eta: bool, available: jnp.ndarray, chunk_size: int
    ):
        """
        params_chunk: pytree with leading axis == chunk_size
        eta_chunk:    (chunk_size, nr_steps, nr_envs)
        available:    scalar int (<= chunk_size) valid rows for the last (possibly short) chunk
        """
        out_flat  = vmapped_chunk(inputs, params_chunk)c # (chunk_size, N, ...)
        out_chunk = out_flat.reshape((chunk_size, nr_steps, nr_envs)) # (chunk_size, S, E)

        def body_fun(t, carry):
            def do_step(carry):
                ldr_t = out_chunk[t]
                eta_t = eta_chunk[t]
                if maximum_eta:
                    step = epsilon / (1.0 + jnp.clip(jnp.max(eta_t), min=0.0))
                else:
                    step = epsilon / (1.0 + jnp.clip(eta_t, min=0.0))
                return (1.0 - step) * carry + step * beta * ldr_t

            return jax.lax.cond(t < available, do_step, lambda c: c, carry)

        return jax.lax.fori_loop(0, chunk_size, body_fun, corr)

    @partial(jax.jit, static_argnames=('chunk_size', 'maximum_eta'))
    def chunked_correct(buffer, inputs, etas, corr, level: jnp.ndarray, chunk_size: int, maximum_eta=maximum_eta):
        """
        buffer: pytree with leading axis >= level (full DiscBuffer)
        inputs:    input to log_density_ratio_fn
        etas:   (>= level, nr_steps, nr_envs)
        corr: initial corrected reward (nr_steps, nr_envs)
        level:  dynamic int32, number of valid params to use (B)
        """
        B = jnp.asarray(level, dtype=jnp.int32)

        n_chunks = (B + chunk_size - 1) // chunk_size
        def loop_body(k, carry):
            corr = carry
            start = k * chunk_size
            available = jnp.asarray(jnp.minimum(chunk_size, B - start), dtype=jnp.int32)

            params_chunk = jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, start_index=start, slice_size=chunk_size),
                buffer
            )
            eta_chunk = jax.lax.dynamic_slice_in_dim(etas, start_index=start, slice_size=chunk_size)

            corr = process_chunk_on_device(
                corr, params_chunk, eta_chunk, inputs,
                maximum_eta=maximum_eta, available=available, chunk_size=chunk_size,
            )
            return corr

        corr = jax.lax.fori_loop(0, n_chunks, loop_body, corr)
        return entropy_coef * jax.lax.stop_gradient(corr) # scaling to reduce critic loss

    return chunked_correct
