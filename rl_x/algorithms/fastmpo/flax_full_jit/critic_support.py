import jax.numpy as jnp


def init_critic_support_state(v_min, v_max, stddev_multiplier):
    center = jnp.asarray((v_min + v_max) / 2.0, dtype=jnp.float32)
    half_range = jnp.asarray((v_max - v_min) / 2.0, dtype=jnp.float32)
    return {
        "center": center,
        "half_range": half_range,
        "target_running_mean": center,
        "target_running_second_moment": jnp.square(center) + jnp.square(half_range / stddev_multiplier),
    }


def update_critic_support(
    state,
    target_mean,
    target_second_moment,
    statistics_rate,
    support_rate,
    stddev_multiplier,
    min_half_range,
    max_half_range,
    epsilon,
):
    target_running_mean = state["target_running_mean"] + statistics_rate * (target_mean - state["target_running_mean"])
    target_running_second_moment = state["target_running_second_moment"] + statistics_rate * (
        target_second_moment - state["target_running_second_moment"]
    )
    target_running_var = jnp.maximum(target_running_second_moment - jnp.square(target_running_mean), epsilon)
    desired_center = target_running_mean
    desired_half_range = jnp.clip(
        stddev_multiplier * jnp.sqrt(target_running_var),
        min_half_range,
        max_half_range,
    )
    center = state["center"] + support_rate * (desired_center - state["center"])
    half_range = state["half_range"] + support_rate * (desired_half_range - state["half_range"])
    return {
        "center": center,
        "half_range": half_range,
        "target_running_mean": target_running_mean,
        "target_running_second_moment": target_running_second_moment,
    }
