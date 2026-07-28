import numpy as np


def init_reward_normalizer_state(nr_envs):
    return {
        "return": np.zeros(nr_envs, dtype=np.float32),
        "previous_done": np.ones(nr_envs, dtype=np.float32),
        "mean": np.zeros((), dtype=np.float32),
        "variance": np.ones((), dtype=np.float32),
        "count": np.asarray(1e-4, dtype=np.float32),
    }


def normalize_reward(state, reward, terminated, truncated, gamma, reward_clip):
    discounted_returns = np.zeros_like(reward)
    previous_return = state["return"]
    previous_done = state["previous_done"]
    for step in range(reward.shape[0]):
        previous_return = reward[step] + (1.0 - previous_done) * gamma * previous_return
        previous_done = np.maximum(terminated[step], truncated[step]).astype(reward.dtype)
        discounted_returns[step] = previous_return
    batch_mean = np.mean(discounted_returns)
    batch_variance = np.var(discounted_returns)
    batch_count = discounted_returns.size
    delta = batch_mean - state["mean"]
    total_count = state["count"] + batch_count
    mean = state["mean"] + delta * batch_count / total_count
    second_moment = state["variance"] * state["count"] + batch_variance * batch_count + delta ** 2 * state["count"] * batch_count / total_count
    variance = second_moment / (total_count - 1.0)
    state = {"return": previous_return, "previous_done": previous_done, "mean": mean, "variance": variance, "count": total_count}
    return state, np.clip(reward / np.sqrt(variance + 1e-8), -reward_clip, reward_clip)
