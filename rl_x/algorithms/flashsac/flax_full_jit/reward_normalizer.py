import jax.numpy as jnp


def init_reward_normalizer_state(nr_envs):
    return {
        "G_r": jnp.zeros((nr_envs,), dtype=jnp.float32),
        "G_r_max": jnp.zeros((), dtype=jnp.float32),
        "rms_mean": jnp.zeros((), dtype=jnp.float32),
        "rms_var": jnp.ones((), dtype=jnp.float32),
        "rms_count": jnp.zeros((), dtype=jnp.float32),
    }


def update_reward_normalizer(state, reward, terminated, truncated, gamma):
    done = jnp.logical_or(terminated, truncated).astype(jnp.float32)
    new_G_r = gamma * (1.0 - done) * state["G_r"] + reward
    new_G_r_max = jnp.maximum(state["G_r_max"], jnp.max(jnp.abs(new_G_r)))
    sample_mean = jnp.mean(new_G_r)
    sample_var = jnp.var(new_G_r)
    sample_count = jnp.float32(new_G_r.shape[0])
    delta = sample_mean - state["rms_mean"]
    total_count = state["rms_count"] + sample_count
    ratio = sample_count / total_count
    new_mean = state["rms_mean"] + delta * ratio
    m_a = state["rms_var"] * (state["rms_count"] + 1e-4)
    m_b = sample_var * sample_count
    m_2 = m_a + m_b + jnp.square(delta) * state["rms_count"] * ratio
    new_var = m_2 / total_count
    return {
        "G_r": new_G_r,
        "G_r_max": new_G_r_max,
        "rms_mean": new_mean,
        "rms_var": new_var,
        "rms_count": total_count,
    }


def normalize_reward(state, reward, normalized_g_max, epsilon=1e-8):
    var_denom = jnp.sqrt(state["rms_var"] + epsilon)
    min_denom = state["G_r_max"] / normalized_g_max
    denom = jnp.maximum(var_denom, min_denom)
    return reward / denom
