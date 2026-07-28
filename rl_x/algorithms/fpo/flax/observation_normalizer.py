import jax.numpy as jnp


def init_observation_normalizer_state(shape):
    return {
        "mean": jnp.zeros(shape, dtype=jnp.float32),
        "var": jnp.ones(shape, dtype=jnp.float32),
        "count": jnp.zeros((), dtype=jnp.float32),
    }


def update_observation_normalizer(state, observations, max_count):
    batch_mean = jnp.mean(observations, axis=0)
    batch_var = jnp.var(observations, axis=0)
    batch_count = jnp.float32(observations.shape[0])
    delta = batch_mean - state["mean"]
    total_count = state["count"] + batch_count
    mean = state["mean"] + delta * batch_count / total_count
    m_a = state["var"] * state["count"]
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + jnp.square(delta) * state["count"] * batch_count / total_count
    update_active = state["count"] < max_count
    return {
        "mean": jnp.where(update_active, mean, state["mean"]),
        "var": jnp.where(update_active, m_2 / total_count, state["var"]),
        "count": jnp.where(update_active, total_count, state["count"]),
    }


def normalize_observation(state, observation, epsilon):
    return (observation - state["mean"]) / (jnp.sqrt(state["var"]) + epsilon)
