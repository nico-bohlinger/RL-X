import jax
import jax.numpy as jnp


def init_reward_normalizer_state(nr_envs):
    return {
        "return": jnp.zeros(nr_envs),
        "previous_done": jnp.ones(nr_envs),
        "mean": jnp.zeros(()),
        "variance": jnp.ones(()),
        "count": jnp.asarray(1e-4),
    }


def normalize_reward(state, reward, terminated, truncated, gamma, reward_clip):
    def discounted_return_step(carry, inputs):
        previous_return, previous_done = carry
        reward_t, terminated_t, truncated_t = inputs
        return_t = reward_t + (1.0 - previous_done) * gamma * previous_return
        done_t = jnp.maximum(terminated_t, truncated_t).astype(reward_t.dtype)
        return (return_t, done_t), return_t

    (last_return, last_done), discounted_returns = jax.lax.scan(discounted_return_step, (state["return"], state["previous_done"]), (reward, terminated, truncated))
    batch_mean = jnp.mean(discounted_returns)
    batch_variance = jnp.var(discounted_returns)
    batch_count = discounted_returns.size
    delta = batch_mean - state["mean"]
    total_count = state["count"] + batch_count
    mean = state["mean"] + delta * batch_count / total_count
    second_moment = state["variance"] * state["count"] + batch_variance * batch_count + delta ** 2 * state["count"] * batch_count / total_count
    variance = second_moment / (total_count - 1.0)
    state = {
        "return": last_return,
        "previous_done": last_done,
        "mean": mean,
        "variance": variance,
        "count": total_count,
    }
    normalized_reward = jnp.clip(reward / jnp.sqrt(variance + 1e-8), -reward_clip, reward_clip)
    return state, normalized_reward
