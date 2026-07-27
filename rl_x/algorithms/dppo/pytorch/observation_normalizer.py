import numpy as np


def init_observation_normalizer_state(shape):
    return {
        "mean": np.zeros(shape, dtype=np.float32),
        "var": np.ones(shape, dtype=np.float32),
        "count": np.array(1e-4, dtype=np.float32),
    }


def update_observation_normalizer(state, observations):
    batch_mean = np.mean(observations, axis=0)
    batch_var = np.var(observations, axis=0)
    batch_count = np.float32(observations.shape[0])
    delta = batch_mean - state["mean"]
    total_count = state["count"] + batch_count
    mean = state["mean"] + delta * batch_count / total_count
    m_a = state["var"] * state["count"]
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * state["count"] * batch_count / total_count
    return {"mean": mean, "var": m_2 / total_count, "count": total_count}


def normalize_observation(state, observation):
    return (observation - state["mean"]) / np.sqrt(state["var"] + 1e-8)
