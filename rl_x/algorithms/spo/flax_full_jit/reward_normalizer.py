import jax
import jax.numpy as jnp


def init_reward_normalizer_state(nr_envs):
    return {
        "return": jnp.zeros(nr_envs),
        "mean": jnp.zeros(nr_envs),
        "variance": jnp.ones(nr_envs),
        "count": jnp.full((nr_envs,), 1e-4),
    }


def normalize_reward(state, reward, terminated, truncated, gamma):
    def discounted_return_step(previous_return, inputs):
        reward_t, terminated_t = inputs
        return_t = reward_t + (1.0 - terminated_t) * gamma * previous_return
        return return_t, return_t

    last_return, discounted_returns = jax.lax.scan(discounted_return_step, state["return"], (reward, terminated))
    batch_mean = jnp.mean(discounted_returns, axis=0)
    batch_variance = jnp.var(discounted_returns, axis=0)
    batch_count = discounted_returns.shape[0]
    delta = batch_mean - state["mean"]
    total_count = state["count"] + batch_count
    mean = state["mean"] + delta * batch_count / total_count
    combined_second_moment = state["variance"] * state["count"] + batch_variance * batch_count + delta ** 2 * state["count"] * batch_count / total_count
    variance = combined_second_moment / total_count
    return {"return": last_return, "mean": mean, "variance": variance, "count": total_count}, jnp.clip(reward / jnp.sqrt(variance[None] + 1e-8), -10.0, 10.0)
