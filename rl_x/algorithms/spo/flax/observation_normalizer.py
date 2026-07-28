import jax.numpy as jnp


def init_observation_normalizer_state(nr_envs, shape):
    return {
        "mean": jnp.zeros((nr_envs,) + shape, dtype=jnp.float32),
        "var": jnp.ones((nr_envs,) + shape, dtype=jnp.float32),
        "count": jnp.full((nr_envs,), 1e-4, dtype=jnp.float32),
    }


def update_observation_normalizer(state, observations):
    delta = observations - state["mean"]
    total_count = state["count"] + 1.0
    mean = state["mean"] + delta / total_count[:, None]
    combined_second_moment = state["var"] * state["count"][:, None] + jnp.square(delta) * state["count"][:, None] / total_count[:, None]
    return {
        "mean": mean,
        "var": combined_second_moment / total_count[:, None],
        "count": total_count,
    }


def normalize_observation(state, observation):
    return jnp.clip((observation - state["mean"]) / jnp.sqrt(state["var"] + 1e-8), -10.0, 10.0)
