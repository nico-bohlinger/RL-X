import jax.numpy as jnp


def get_evaluation_termination_counts(env_state):
    termination_counts = {
        "episode_end_count": jnp.sum(env_state.terminated | env_state.truncated),
        "termination_count": jnp.sum(env_state.terminated),
        "truncation_count": jnp.sum(env_state.truncated),
    }
    for info_key in (
        "env_info/termination_below_height",
        "env_info/termination_velocity_saturation",
        "env_info/termination_nonfinite_state",
    ):
        if info_key in env_state.info:
            termination_counts[info_key] = jnp.sum(env_state.info[info_key])
    return termination_counts


def get_evaluation_termination_metrics(termination_counts):
    episode_end_count = termination_counts["episode_end_count"]
    episode_end_denominator = jnp.maximum(episode_end_count, 1)
    metrics = {
        "eval/episode_end_count": episode_end_count,
        "eval/termination_rate": termination_counts["termination_count"] / episode_end_denominator,
        "eval/truncation_rate": termination_counts["truncation_count"] / episode_end_denominator,
    }
    for info_key, metric_key in {
        "env_info/termination_below_height": "eval/termination_below_height_rate",
        "env_info/termination_velocity_saturation": "eval/termination_velocity_saturation_rate",
        "env_info/termination_nonfinite_state": "eval/termination_nonfinite_state_rate",
    }.items():
        if info_key in termination_counts:
            metrics[metric_key] = termination_counts[info_key] / episode_end_denominator
    return metrics
