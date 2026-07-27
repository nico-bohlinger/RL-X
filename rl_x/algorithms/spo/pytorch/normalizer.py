import numpy as np


def update_observation_normalizer(state, observations):
    delta = observations - state["mean"]
    total_count = state["count"] + 1.0
    mean = state["mean"] + delta / total_count[:, None]
    combined_second_moment = state["var"] * state["count"][:, None] + np.square(delta) * state["count"][:, None] / total_count[:, None]
    return {"mean": mean, "var": combined_second_moment / total_count[:, None], "count": total_count}


def normalize_observation(state, observation):
    return np.clip((observation - state["mean"]) / np.sqrt(state["var"] + 1e-8), -10.0, 10.0)


def normalize_reward(state, reward, terminated, gamma):
    previous_return = state["return"]
    discounted_returns = np.zeros_like(reward)
    for step in range(reward.shape[0]):
        previous_return = reward[step] + (1.0 - terminated[step]) * gamma * previous_return
        discounted_returns[step] = previous_return
    batch_mean = np.mean(discounted_returns, axis=0)
    batch_variance = np.var(discounted_returns, axis=0)
    batch_count = discounted_returns.shape[0]
    delta = batch_mean - state["mean"]
    total_count = state["count"] + batch_count
    mean = state["mean"] + delta * batch_count / total_count
    combined_second_moment = state["variance"] * state["count"] + batch_variance * batch_count + delta ** 2 * state["count"] * batch_count / total_count
    variance = combined_second_moment / total_count
    return {"return": previous_return, "mean": mean, "variance": variance, "count": total_count}, np.clip(reward / np.sqrt(variance[None] + 1e-8), -10.0, 10.0)
