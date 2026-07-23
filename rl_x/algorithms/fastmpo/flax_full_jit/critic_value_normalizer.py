import jax.numpy as jnp


def init_critic_value_normalizer_state(nr_envs):
    return {
        "discounted_return": jnp.zeros((nr_envs,), dtype=jnp.float32),
        "discounted_return_abs_max": jnp.zeros((), dtype=jnp.float32),
        "running_mean": jnp.zeros((), dtype=jnp.float32),
        "running_var": jnp.ones((), dtype=jnp.float32),
        "count": jnp.zeros((), dtype=jnp.float32),
    }


def update_critic_value_normalizer(state, reward, terminated, truncated, gamma):
    done = jnp.logical_or(terminated, truncated).astype(jnp.float32)
    discounted_return = gamma * (1.0 - done) * state["discounted_return"] + reward
    discounted_return_abs_max = jnp.maximum(state["discounted_return_abs_max"], jnp.max(jnp.abs(discounted_return)))
    batch_mean = jnp.mean(discounted_return)
    batch_var = jnp.var(discounted_return)
    batch_count = jnp.float32(discounted_return.shape[0])
    total_count = state["count"] + batch_count
    delta = batch_mean - state["running_mean"]
    running_mean = state["running_mean"] + delta * batch_count / total_count
    m_a = state["running_var"] * state["count"]
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + jnp.square(delta) * state["count"] * batch_count / total_count
    running_var = m_2 / total_count
    return {
        "discounted_return": discounted_return,
        "discounted_return_abs_max": discounted_return_abs_max,
        "running_mean": running_mean,
        "running_var": running_var,
        "count": total_count,
    }


def get_critic_value_scale(state, normalized_return_max, epsilon):
    return jnp.maximum(
        jnp.sqrt(state["running_var"] + epsilon),
        state["discounted_return_abs_max"] / normalized_return_max,
    )
