import jax
import jax.numpy as jnp


def build_zeta_cdf(mu, max_n):
    ns = jnp.arange(1, max_n + 1, dtype=jnp.float32)
    pmf = ns ** (-mu)
    pmf = pmf / jnp.sum(pmf)
    return jnp.cumsum(pmf)


def sample_zeta(key, cdf):
    u = jax.random.uniform(key, shape=())
    return jnp.argmax((u < cdf).astype(jnp.int32)) + 1


def init_noise_state(nr_envs, action_dim, key):
    return {
        "noise": jax.random.normal(key, shape=(nr_envs, action_dim), dtype=jnp.float32),
        "count": jnp.zeros((), dtype=jnp.int32),
        "n": jnp.ones((), dtype=jnp.int32),
    }


def step_noise(state, key, cdf):
    noise_key, n_key = jax.random.split(key)
    new_noise = jax.random.normal(noise_key, shape=state["noise"].shape, dtype=jnp.float32)
    new_n = sample_zeta(n_key, cdf)

    reinit = (state["count"] == 0) | (state["count"] >= state["n"])
    noise = jnp.where(reinit, new_noise, state["noise"])
    n = jnp.where(reinit, new_n, state["n"])
    count = jnp.where(reinit, jnp.zeros_like(state["count"]), state["count"]) + 1
    return {"noise": noise, "count": count, "n": n}
