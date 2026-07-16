import jax.numpy as jnp


def init_reward_normalizer_state(nr_envs):
    return {
        "G_r": jnp.zeros((nr_envs,), dtype=jnp.float32),
        "rms_mean": jnp.zeros((), dtype=jnp.float32),
        "rms_var": jnp.ones((), dtype=jnp.float32),
        "rms_count": jnp.array(1e-4, dtype=jnp.float32),
    }


def update_reward_normalizer(state, reward, terminated, truncated, gamma):
    done = jnp.logical_or(terminated, truncated).astype(jnp.float32)
    new_G_r = gamma * (1.0 - done) * state["G_r"] + reward
    sample_mean = jnp.mean(new_G_r)
    sample_var = jnp.var(new_G_r)
    sample_count = jnp.float32(new_G_r.shape[0])
    delta = sample_mean - state["rms_mean"]
    total_count = state["rms_count"] + sample_count
    ratio = sample_count / total_count
    new_mean = state["rms_mean"] + delta * ratio
    m_a = state["rms_var"] * state["rms_count"]
    m_b = sample_var * sample_count
    m_2 = m_a + m_b + jnp.square(delta) * state["rms_count"] * ratio
    new_var = m_2 / total_count
    return {
        "G_r": new_G_r,
        "rms_mean": new_mean,
        "rms_var": new_var,
        "rms_count": total_count,
    }


def normalize_reward(state, reward, epsilon=1e-8):
    return reward / (jnp.sqrt(state["rms_var"]) + epsilon)
